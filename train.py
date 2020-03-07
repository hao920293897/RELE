import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
from sklearn.metrics import average_precision_score
import sys
import ctypes
import threading
from tensorflow.python import debug as tf_debug
os.environ["CUDA_VISIBLE_DEVICES"]="1"
export_path = '/home/zlh/beifen/RE/gids_clean/'
word_vec = np.load(export_path + 'vec.npy')
f = open(export_path + "config", 'r')
config = json.loads(f.read())
f.close()


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nbatch_kg',100,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.01,'learning rate for kg')
tf.app.flags.DEFINE_integer('ent_total',config['entity_total'],'total of entities')

tf.app.flags.DEFINE_integer('katt_flag',0, '1 for katt, 0 for att')
tf.app.flags.DEFINE_integer('desc_flag',0,'1 for desc,0 for none')
tf.app.flags.DEFINE_boolean('desc_att',True,'true for desc_katt, false for none')
tf.app.flags.DEFINE_float('alpha',2.6,'weights')
tf.app.flags.DEFINE_string('save_name','cnn-att','the name of saved model')
tf.app.flags.DEFINE_integer('gate',1,'gate')
tf.app.flags.DEFINE_integer('merge_flag',0,'1 for merge desc,0 for none')
tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', config['textual_rel_total'],'maximum of relations')
tf.app.flags.DEFINE_integer('description_length',config['desc_len'],'max number of words in one description of an entity')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch',30,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',160,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.5,'learning rate for nn')

tf.app.flags.DEFINE_float('weight_decay',0.0001,'weight_decay')
tf.app.flags.DEFINE_float('keep_prob',0.5,'dropout rate')

tf.app.flags.DEFINE_string('model_dir','./model','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')


def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def make_shape(array,last_dim):
	output = []
	for i in array:
		for j in i:
			output.append(j)
	output = np.array(output)
	if np.shape(output)[-1]==last_dim:
		return output
	else:
		print ('Make Shape Error!')

def main(_):

	print ('reading word embedding')
	word_vec = np.load(export_path + 'vec.npy')
	print('reading entity embedding')
	ent_embedding = np.load(export_path + 'ent_embedding.npy')
	print('reading relation embedding')
	rel_embedding = np.load(export_path + 'rel_embedding.npy')
	print ('reading training data')

	instance_triple = np.load(export_path + 'train_instance_triple.npy')
	instance_scope = np.load(export_path + 'train_instance_scope.npy')
	train_len = np.load(export_path + 'train_len.npy')
	train_label = np.load(export_path + 'train_label.npy')
	train_word = np.load(export_path + 'train_word.npy')
	train_pos1 = np.load(export_path + 'train_pos1.npy')
	train_pos2 = np.load(export_path + 'train_pos2.npy')
	train_mask = np.load(export_path + 'train_mask.npy')
	train_head = np.load(export_path + 'train_head.npy')
	train_tail = np.load(export_path + 'train_tail.npy')
	train_desc_tail = np.load(export_path + 'train_desc_tail.npy')
	train_desc_head = np.load(export_path + 'train_desc_head.npy')
	desc_all = np.load(export_path+'desc_all.npy')

	print ('reading finished')
	print ('mentions 		: %d' % (len(instance_triple)))
	print ('sentences		: %d' % (len(train_len)))
	print ('relations		: %d' % (FLAGS.num_classes))
	print ('word size		: %d' % (len(word_vec[0])))
	print ('position size 	: %d' % (FLAGS.pos_size))
	print ('hidden size		: %d' % (FLAGS.hidden_size))
	reltot = {}
	for index, i in enumerate(train_label):
		if not i in reltot:
			reltot[i] = 1.0
		else:
			reltot[i] += 1.0
	for i in reltot:
		reltot[i] = 1/(reltot[i] ** (0.05))
	print ('building network...')
	i=tf.ConfigProto()
	i.gpu_options.allow_growth = True
	sess = tf.Session(config=i)

	if FLAGS.model.lower() == "cnn":
		model = network.CNN(is_training = True, word_embeddings = word_vec,ent_embedding=ent_embedding,rel_embedding=rel_embedding)

	global_step = tf.Variable(0,name='global_step',trainable=False)
	global_step_kg = tf.Variable(0,name='global_step_kg',trainable=False)
	tf.summary.scalar('learning_rate', FLAGS.learning_rate)
	tf.summary.scalar('learning_rate_kg', FLAGS.learning_rate_kg)
	sess.run(tf.global_variables_initializer())

	optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
	grads_and_vars = optimizer.compute_gradients(model.loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)


	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

	saver = tf.train.Saver(max_to_keep=None)

	print ('building finished')

	def train_nn(coord):
		def train_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope, weights,head_desc,tail_desc):
			feed_dict = {
				model.head_index: head,
				model.tail_index: tail,
				model.word: word,
				model.pos1: pos1,
				model.pos2: pos2,
				model.mask: mask,
				model.len : leng,
				model.label_index: label_index,
				model.label: label,
				model.scope: scope,
				model.keep_prob: FLAGS.keep_prob,
				model.weights: weights,
				model.head_description:head_desc,
				model.tail_description:tail_desc
			}
			_, step, loss, summary, output, correct_predictions = sess.run([train_op, global_step, model.loss, merged_summary, model.output, model.correct_predictions], feed_dict)
			summary_writer.add_summary(summary, step)
			return output, loss, correct_predictions

		stack_output = []
		stack_label = []
		stack_ce_loss = []

		train_order = list(range(len(instance_triple)))

		save_epoch = 2
		eval_step = 300

		for one_epoch in range(FLAGS.max_epoch):
		# one_epoch = 0
		# while not coord.should_stop():
			print('epoch '+str(one_epoch+1)+' starts!')
			# one_epoch += 1
			np.random.shuffle(train_order)
			s1 = 0.0
			s2 = 0.0
			tot1 = 0.0
			tot2 = 0.0
			losstot = 0.0
			for i in range(int(len(train_order)/float(FLAGS.batch_size))):
				## randomly sample a batch of input scope
				input_scope = np.take(instance_scope, train_order[i * FLAGS.batch_size:(i+1)*FLAGS.batch_size], axis=0)
				index = []
				scope = [0]
				label = [] ## relation label corresponding to each scope
				weights = []
				for num in input_scope:
					index = index + list(range(num[0], num[1] + 1))
					label.append(train_label[num[0]])
					if train_label[num[0]] > 53:
						print (train_label[num[0]])
					scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)
					weights.append(reltot[train_label[num[0]]])
				label_ = np.zeros((FLAGS.batch_size, FLAGS.num_classes))
				label_[np.arange(FLAGS.batch_size), label] = 1
				output, loss, correct_predictions = train_step(train_head[index], train_tail[index], train_word[index,:], train_pos1[index,:], train_pos2[index,:], train_mask[index,:], train_len[index],train_label[index], label_, np.array(scope),
															   weights,train_desc_head[index],train_desc_tail[index])
				num = 0
				s = 0
				losstot += loss
				for num in correct_predictions:
					if label[s] == 0:
						tot1 += 1.0
						if num:
							s1+= 1.0
					else:
						tot2 += 1.0
						if num:
							s2 += 1.0
					s = s + 1

				time_str = datetime.datetime.now().isoformat()
				print ("batch %d step %d time %s | loss : %f, NA accuracy: %f, not NA accuracy: %f" % (one_epoch, i, time_str, loss, s1 / tot1, s2 / tot2))
				current_step = tf.train.global_step(sess, global_step)

			if (one_epoch+1) % save_epoch == 0:
				print ('epoch '+str(one_epoch+1)+' has finished')
				print ('saving model...')
				path = saver.save(sess,FLAGS.model_dir+FLAGS.save_name+'/'+FLAGS.model+str(FLAGS.katt_flag), global_step=current_step)
				print ('have savde model to '+path)

		coord.request_stop()


	coord = tf.train.Coordinator()
	threads = []
	threads.append(threading.Thread(target=train_nn, args=(coord,)))
	for t in threads: t.start()
	coord.join(threads)

if __name__ == "__main__":
	tf.app.run()
