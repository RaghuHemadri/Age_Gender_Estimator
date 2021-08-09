import os
import cv2
import dlib
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--input_path", type=str, default=None, help="input video path")
    parser.add_argument("--output_path", type=str, default=None, help="output video path")
    args = parser.parse_args()
    return args

def main(sess,age,gender,train_mode,images_pl,in_path,out_path):
    args = get_args()
    #in_path = args.input_path
    #out_path = args.output_path
    depth = args.depth
    k = args.width
    img_size = 160

    # for face detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    # load model and weights
    cap = cv2.VideoCapture(in_path)
    ret, frame = cap.read()
    height, width, channels = frame.shape
    frameSize = (height,width)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path, fourcc, fps, frameSize)
    cap.release()
    cap = cv2.VideoCapture(in_path)

    while (ret == True):
        # get video frame
        ret, img = cap.read()
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        if ret ==True:
	        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        img_h, img_w, _ = np.shape(input_img)

	        # detect faces using dlib detector
	        detected = detector(input_img, 1)
	        faces = np.empty((len(detected), img_size, img_size, 3))

	        for i, d in enumerate(detected):
	            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
	            # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
	        #
	        if len(detected) > 0:
	            # predict ages and genders of the detected faces
	            ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})

	        # draw results
	        for i, d in enumerate(detected):
	            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
	            xw1 = max(int(x1 - 0.4 * w), 0)
	            yw1 = max(int(y1 - 0.4 * h), 0)
	            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
	            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
	            if int(ages[i])<22 and int(genders[i]) == 0:
	                cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 10)
	                cv2.putText(img, 'minor', (int(xw1-0.2*w), int(yw1-0.2*h)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), cv2.LINE_AA)

        out.write(img)

    cap.release()
    out.release()

def load_network(model_path):
    #with tf.Graph().as_default():
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore model!")
        else:
            pass
        return sess,age,gender,train_mode,images_pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")
    parser.add_argument("--input_path", type=str, default=None, help="input video path")
    parser.add_argument("--output_path", type=str, default=None, help="output video path")
    args = parser.parse_args()
    sess, age, gender, train_mode,images_pl = load_network(args.model_path)
    main(sess,age,gender,train_mode,images_pl, args.input_path, args.output_path)