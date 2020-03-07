import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import json
import sys
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import ctypes
from tensorflow.python import debug as tf_debug

os.environ["CUDA_VISIBLE_DEVICES"]="1"
export_path =  "/home/zlh/beifen/RE/gids_clean/"
word_vec = np.load(export_path + 'vec.npy')
f = open(export_path + "config", 'r')
config = json.loads(f.read())
f.close()

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nbatch_kg',100,'entity numbers used each training time')
tf.app.flags.DEFINE_float('margin',1.0,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate_kg',0.1,'learning rate for kg')
tf.app.flags.DEFINE_integer('ent_total',config['entity_total'],'total of entities')

tf.app.flags.DEFINE_integer('katt_flag', 0, '1 for katt, 0 for att')
tf.app.flags.DEFINE_integer('merge_flag',0,'1 for merge desc,0 for none')

tf.app.flags.DEFINE_string('model', 'cnn', 'neural models to encode sentences')
tf.app.flags.DEFINE_integer('max_length',config['fixlen'],'maximum of number of words in one sentence')
tf.app.flags.DEFINE_integer('pos_num', config['maxlen'] * 2 + 1,'number of position embedding vectors')
tf.app.flags.DEFINE_integer('num_classes', config['textual_rel_total'],'maximum of relations')
tf.app.flags.DEFINE_integer('description_length', 100,'max number of words in one description of an entity')
tf.app.flags.DEFINE_float('alpha',2.6,'weights')
tf.app.flags.DEFINE_string('save_name','cnn-att','the name of saved model')
tf.app.flags.DEFINE_integer('desc_flag',0,'1 for desc,0 for none')
tf.app.flags.DEFINE_boolean('desc_att',True,'true for desc_katt, false for none')
tf.app.flags.DEFINE_integer('gate',1,'gate')

tf.app.flags.DEFINE_integer('hidden_size',230,'hidden feature size')
tf.app.flags.DEFINE_integer('pos_size',5,'position embedding size')

tf.app.flags.DEFINE_integer('max_epoch',20,'maximum of training epochs')
tf.app.flags.DEFINE_integer('batch_size',262,'entity numbers used each training time')
tf.app.flags.DEFINE_float('learning_rate',0.01,'entity numbers used each training time')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')

tf.app.flags.DEFINE_float('keep_prob',1.0,'dropout rate')
# tf.app.flags.DEFINE_integer('test_batch_size',263,'entity numbers used each test time')

tf.app.flags.DEFINE_integer('test_batch_size',262,'entity numbers used each test time')
tf.app.flags.DEFINE_string('checkpoint_path','./model','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')


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

def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def precision_at_k(yhat_raw, y, k):
	# num true labels in top k predictions / k
	sortd = np.argsort(yhat_raw)[:, ::-1]
	topk = sortd[:, :k]

	# get precision at k for each example
	vals = []
	for i, tk in enumerate(topk):
		if len(tk) > 0:
			num_true_in_top_k = y[i, tk].sum()
			denom = len(tk)
			vals.append(num_true_in_top_k / float(denom))

	return np.mean(vals)


def intersect_size(yhat, y, axis):
	# axis=0 for label-level union (macro). axis=1 for instance-level
	return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_precision(yhat, y):
	num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
	return np.mean(num)


def macro_recall(yhat, y):
	num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
	return np.mean(num)


def macro_f1(yhat, y):
	prec = macro_precision(yhat, y)
	rec = macro_recall(yhat, y)
	if prec + rec == 0:
		f1 = 0.
	else:

		f1 = 2 * (prec * rec) / (prec + rec)
	return f1

def main(_):

	print ('reading word embedding')
	word_vec = np.load(export_path + 'vec.npy')
	print('reading entity embedding')
	ent_embedding = np.load(export_path + 'ent_embedding.npy')
	print('reading relation embedding')
	rel_embedding = np.load(export_path + 'rel_embedding.npy')
	print ('reading test data')
	test_instance_triple = np.load(export_path + 'test_instance_triple.npy')
	test_instance_scope = np.load(export_path + 'test_instance_scope.npy')
	test_len = np.load(export_path + 'test_len.npy')
	test_label = np.load(export_path +'test_label.npy')
	test_word = np.load(export_path + 'test_word.npy')
	test_pos1 = np.load(export_path + 'test_pos1.npy')
	test_pos2 = np.load(export_path + 'test_pos2.npy')
	test_mask = np.load(export_path + 'test_mask.npy')
	test_head = np.load(export_path + 'test_head.npy')
	test_tail = np.load(export_path + 'test_tail.npy')
	train_desc_tail = np.load(export_path + 'test_desc_tail.npy')
	train_desc_head = np.load(export_path + 'test_desc_head.npy')
	print ('reading finished')
	print ('mentions 		: %d' % (len(test_instance_triple)))
	print ('sentences		: %d' % (len(test_len)))
	print ('relations		: %d' % (FLAGS.num_classes))
	print ('word size		: %d' % (len(word_vec[0])))
	print ('position size 	: %d' % (FLAGS.pos_size))
	print ('hidden size		: %d' % (FLAGS.hidden_size))
	print ('reading finished')
	# desc = {}
	# with open(export_path + 'desc.txt') as f:
	# 	for content in f:
	# 		en_id, en_desc = content.strip().split('\t')
	# 		en_desc = en_desc.strip().split(',')
	# 		en_desc = [int(word) for word in en_desc]
	# 		desc[int(en_id)] = en_desc
	print ('building network...')
	sess_db = tf.Session()
	# sess_db = tf_debug.LocalCLIDebugWrapperSession(sess)
	# sess_db.add_tensor_filter('has_inf_or_nan',tf_debug.has_inf_or_nan)
	merged_summary = tf.summary.merge_all()
	global_step = tf.Variable(0,name='global_step',trainable=False)
	if FLAGS.model.lower() == "cnn":
		model = network.CNN(is_training = False, word_embeddings = word_vec,ent_embedding=ent_embedding,rel_embedding=rel_embedding)
	elif FLAGS.model.lower() == "pcnn":
		model = network.PCNN(is_training = False, word_embeddings = word_vec,ent_embedding=ent_embedding,rel_embedding=rel_embedding)
	elif FLAGS.model.lower() == "lstm":
		model = network.RNN(is_training = False, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "gru":
		model = network.RNN(is_training = False, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
		model = network.BiRNN(is_training = False, word_embeddings = word_vec, cell_name = "LSTM", simple_position = True)
	elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
		model = network.BiRNN(is_training = False, word_embeddings = word_vec, cell_name = "GRU", simple_position = True)
	sess_db.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	def test_step(head, tail, word, pos1, pos2, mask, leng, label_index, label, scope,head_desc,tail_desc):
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
			model.head_description: head_desc,
			model.tail_description: tail_desc
		}

		if FLAGS.katt_flag == 1:
			output ,head_desc_att, tail_desc_att = sess_db.run([model.test_output,model.head_desc_att,model.tail_desc_att], feed_dict)
		else:
			output = sess_db.run(model.test_output, feed_dict)

		# np.save('./case_study/head_desc_att',head_desc_att)
		# np.save('./case_study/tail_desc_att',tail_desc_att)
		# output = sess_db.run(model.test_output, feed_dict)
		return output

	f = open('results.txt','w')
	f.write('iteration\taverage precision\tP@100\tP@300\tP@500\n')
	for iters in range(1,15):
		print (iters)
		saver.restore(sess_db, FLAGS.checkpoint_path +FLAGS.save_name+'/'+ FLAGS.model+str(FLAGS.katt_flag)+"-"+str(80 *iters))
		summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess_db.graph)
		stack_output = []
		stack_label = []

		iteration = len(test_instance_scope)//FLAGS.test_batch_size

		for i in range(iteration):
			temp_str= 'running '+str(i)+'/'+str(iteration)+'...'
			sys.stdout.write(temp_str+'\r')
			sys.stdout.flush()
			input_scope = test_instance_scope[i * FLAGS.test_batch_size:(i+1)*FLAGS.test_batch_size]
			index = []
			scope = [0]
			label = []
			# print('input_scope:',input_scope)
			for num in input_scope:
				index = index + list(range(num[0], num[1] + 1))
				label.append(test_label[num[0]])
				scope.append(scope[len(scope)-1] + num[1] - num[0] + 1)

			label_ = np.zeros((FLAGS.test_batch_size, FLAGS.num_classes))
			label_[np.arange(FLAGS.test_batch_size), label] = 1
			output = test_step(test_head[index], test_tail[index], test_word[index,:], test_pos1[index,:], test_pos2[index,:], test_mask[index,:], test_len[index], test_label[index], label_, np.array(scope),
							   train_desc_head[index], train_desc_tail[index])
			stack_output.append(output)
			stack_label.append(label_)

		# print('attention score:',np.shape(attention_score))
		# np.save('attention_scpre',attention_score)
		print ('evaluating...')
		# print(stack_output)


		# ff = open('attention.txt','w')
		# ff.write(attention_score)
		# ff.close()
		stack_output = np.concatenate(stack_output, axis=0)
		stack_label = np.concatenate(stack_label, axis = 0)

		exclude_na_flatten_output = stack_output[:,1:]
		exclude_na_flatten_label = stack_label[:,1:]
		print (exclude_na_flatten_output.shape)
		print (exclude_na_flatten_label.shape)

		# print (exclude_na_flatten_output)


		np.save('./'+'model'+str(FLAGS.alpha)+'/'+FLAGS.model+'+sen_att_all_prob_'+str(iters)+'.npy', exclude_na_flatten_output)
		np.save('./'+'model'+str(FLAGS.alpha)+'/'+FLAGS.model+'+sen_att_all_label_'+str(iters)+'.npy',exclude_na_flatten_label)


		average_precision = average_precision_score(exclude_na_flatten_label,exclude_na_flatten_output, average = "micro")
		exclude_na_flatten_label = np.reshape(exclude_na_flatten_label,-1)
		exclude_na_flatten_output = np.reshape(exclude_na_flatten_output,-1)
		order = np.argsort(-exclude_na_flatten_output)
		p_100 = np.mean(exclude_na_flatten_label[order[:100]])
		p_300 = np.mean(exclude_na_flatten_label[order[:300]])
		p_500 = np.mean(exclude_na_flatten_label[order[:500]])
		print ('pr: '+str(average_precision))
		print('p@100:' + str(p_100))
		print('p@300:' + str(p_300))
		print('p@500:' + str(p_500))

		f.write(str(average_precision)+'\t'+str(p_100)+'\t'+str(p_300)+'\t'+str(p_500)+'\n')
	f.close()

if __name__ == "__main__":
	tf.app.run()
