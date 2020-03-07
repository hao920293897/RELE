#coding=utf-8
import numpy as np
import os
import json
import re
from nltk import sent_tokenize


# folder of training datasets
data_path = "/home/zlh/beifen/RE/origin_gids/"
# files to export data
export_path = "/home/zlh/beifen/RE/gids_clean/"
#length of sentence
fixlen = 120
#max length of position embedding is 100 (-100~+100)
maxlen = 100
#max length of entity description
desc_length = 100

word2id = {}
relation2id = {}
entity2desc = {}
word_size = 0
word_vec = None

def pos_embed(x):
	return max(0, min(x + maxlen, maxlen + maxlen + 1))

def find_index(x,y):
	for index, item in enumerate(y):
		if x == item:
			return index
	return -1

def init_word():
	# reading word embedding data...
	global word2id, word_size
	res = []
	ff = open(export_path + "/entity2id.txt", "w")
	f = open(data_path + "kg/train.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h, t, r = content.strip().split("\t")
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	f = open(data_path + "text_clean/train.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h,t = content.strip().split("\t")[:2]
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	f = open(data_path + "text_clean/test.txt", "r")
	while True:
		content = f.readline()
		if content == "":
			break
		h,t = content.strip().split("\t")[:2]
		if not h in word2id:
			word2id[h] = len(word2id)
			ff.write("%s\t%d\n"%(h, word2id[h]))
		if not t in word2id:
			word2id[t] = len(word2id)
			ff.write("%s\t%d\n"%(t, word2id[t]))
	f.close()
	res.append(len(word2id))
	ff.close()

	print ('reading word embedding data...')
	f = open(data_path + 'text/vec.txt', "r")
	total, size = f.readline().strip().split()[:2]
	total = (int)(total)
	word_size = (int)(size)
	vec = np.ones((total + res[0], word_size), dtype = np.float32)
	for i in range(total):
		content = f.readline().strip().split()
		word2id[content[0]] = len(word2id)
		for j in range(word_size):
			vec[i + res[0]][j] = (float)(content[j+1])
	f.close()
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	global word_vec
	word_vec = vec
	res.append(len(word2id))
	return res

def init_relation():
	# reading relation ids...
	global relation2id
	print ('reading relation ids...')
	res = []
	ff = open(export_path + "/relation2id.txt", "w")
	f = open(data_path + "text/relation2id.txt","r")
	total = (int)(f.readline().strip())
	for i in range(total):
		content = f.readline().strip().split()
		if not content[0] in relation2id:
			relation2id[content[0]] = len(relation2id)
			ff.write("%s\t%d\n"%(content[0], relation2id[content[0]]))
	f.close()
	res.append(len(relation2id))
	f = open(data_path + "kg/train.txt", "r")
	for i in f.readlines():
		h, t, r = i.strip().split("\t")
		if not r in relation2id:
			relation2id[r] = len(relation2id)
			ff.write("%s\t%d\n"%(r, relation2id[r]))
	f.close()
	ff.close()
	res.append(len(relation2id))
	return res

##按相同实体对（h,t）对数据进行排序，即（h,t）相同的三元组放在一起
def sort_files(name, limit):
	hash = {}
	f = open(data_path + "text_clean/" + name + '.txt','r')
	s = 0
	while True:
		content = f.readline()
		if content == '':
			break
		s = s + 1
		origin_data = content
		content = content.strip().split()
		en1_id = content[0]
		en2_id = content[1]
		rel_name = content[4]
		##if the relation does not exist in train dataset,the relation_id is NA 
		if (rel_name in relation2id) and ((int)(relation2id[rel_name]) < limit[0]):
			relation = relation2id[rel_name]
		else:
			relation = relation2id['NA']
		id1 = str(en1_id)+"#"+str(en2_id)
		id2 = str(relation)
		if not id1 in hash:
			hash[id1] = {}
		if not id2 in hash[id1]:
			hash[id1][id2] = []
		hash[id1][id2].append(origin_data)
	f.close()
	f = open(data_path + name + "_sort.txt", "w")
	f.write("%d\n"%(s))
	for i in hash:
		for j in hash[i]:
			for k in hash[i][j]:
				f.write(k)
	f.close()

def init_entity_description(entity_total):
	global entity2desc
	ff  = open(export_path+'desc.txt','w')
	desc = np.zeros([entity_total,desc_length])
	with open(data_path+'text/enmid2desc.txt','r') as f:
		for content in f :
			en_mid,en_desc = content.strip().split('\t')
			en_desc_l = [word2id[word] if word in word2id else word2id['UNK'] for word in en_desc.split()]
			if len(en_desc_l) > desc_length:
				en_desc_l = en_desc_l[:desc_length]
			elif len(en_desc_l) < desc_length:
				for i in range(len(en_desc_l),desc_length):
					en_desc_l.append(word2id['BLANK'])
			entity2desc[word2id[en_mid]] = en_desc_l
			desc[word2id[en_mid]] = en_desc_l
			ff.write('%d\t%s\n'%(word2id[en_mid],str(en_desc_l)))
	ff.close()
	return desc,len(entity2desc)

def init_train_files(name, limit):
	print ('reading ' + name +' data...')
	f = open(data_path + name + '.txt','r')
	total = (int)(f.readline().strip())
	sen_word = np.zeros((total, fixlen), dtype = np.int32) #word id sequence of the sentence
	sen_pos1 = np.zeros((total, fixlen), dtype = np.int32) #position embedding of head entity 
	sen_pos2 = np.zeros((total, fixlen), dtype = np.int32) #position embedding of tail entity
	sen_mask = np.zeros((total, fixlen), dtype = np.int32) #relative position mask sequence of the sentence
	sen_len = np.zeros((total), dtype = np.int32) #length of sentence (<=max len)
	sen_label = np.zeros((total), dtype = np.int32) #relation label corresponding to the sentence
	sen_head = np.zeros((total), dtype = np.int32)  #head entity mention of the sentence
	sen_tail = np.zeros((total), dtype = np.int32)  #tail entity mention of the sentence
	instance_scope = [] ## element[index1,index2] ,the index scope of the sentences instance that mention the same triple (h,r,t)
	instance_triple = [] ## element (h,r,t),the sets of instance triple without repeat
	desc_head = np.zeros((total,desc_length),dtype=np.int32)
	desc_tail = np.zeros((total,desc_length),dtype=np.int32)
	for s in range(total):
		content = f.readline().strip().split()
		sentence = content[5:-1]
		en1_id = content[0]
		en2_id = content[1]
		en1_name = content[2]
		en2_name = content[3]
		rel_name = content[4]
		en1_desc = entity2desc.get(word2id[en1_id])
		en2_desc = entity2desc.get(word2id[en2_id])
		if en1_desc is not None:
			desc_head[s] = en1_desc
		if en2_desc is not  None:
			desc_tail[s] = en2_desc
		##若关系为训练集中存在的关系，映射为对应的relation_id；否则映射为未知关系“NA”的id
		if rel_name in relation2id and ((int)(relation2id[rel_name]) < limit[0]):
			relation = relation2id[rel_name]
		else:
			relation = relation2id['NA']
		en1pos = 0 ##实体1的位置
		en2pos = 0 ##实体2的位置
		for i in range(len(sentence)):
			if sentence[i] == en1_name:
				sentence[i] = en1_id  ##句子中的entity name替换为entity id
				en1pos = i ##记录句中位置
				sen_head[s] = word2id[en1_id] ##记录该句子中的头实体（映射为其word id），
			if sentence[i] == en2_name:
				sentence[i] = en2_id
				en2pos = i
				sen_tail[s] = word2id[en2_id]
		en_first = min(en1pos,en2pos) #(h,t)位置最小值
		en_second = en1pos + en2pos - en_first #(h,t)位置最大值
		for i in range(fixlen):
			sen_word[s][i] = word2id['BLANK']
			sen_pos1[s][i] = pos_embed(i - en1pos)
			sen_pos2[s][i] = pos_embed(i - en2pos)
			if i >= len(sentence):#句长不够，剩余位置标记为0
				sen_mask[s][i] = 0
			elif i - en_first<=0: ##en_first位置前的单词 标记为1
				sen_mask[s][i] = 1
			elif i - en_second<=0: ##en_first和en_second位置中间的单词标记为2
				sen_mask[s][i] = 2
			else:
				sen_mask[s][i] = 3 ##en_second位置之后的单词标记为3
		
		#convert word sequence of the sentence to word id sequence
		for i, word in enumerate(sentence):
			if i >= fixlen:
				break
			elif not word in word2id:
				sen_word[s][i] = word2id['UNK']
			else:
				sen_word[s][i] = word2id[word]
		sen_len[s] = min(fixlen, len(sentence))
		sen_label[s] = relation
		#put the same entity pair sentences into a dict
		tup = (en1_id,en2_id,relation)
		if instance_triple == [] or instance_triple[len(instance_triple) - 1] != tup:
			instance_triple.append(tup)
			instance_scope.append([s,s])
		instance_scope[len(instance_triple) - 1][1] = s
		if (s+1) % 100 == 0:
			print (s)
	return np.array(instance_triple), np.array(instance_scope), sen_len, sen_label, sen_word, sen_pos1, sen_pos2, sen_mask, sen_head, sen_tail,desc_head,desc_tail

def init_kg():
	ff = open(export_path + "/triple2id.txt", "w")
	f = open(data_path + "kg/train.txt", "r")
	content = f.readlines()
	ff.write("%d\n"%(len(content)))
	for i in content:
		h,t,r = i.strip().split("\t")
		ff.write("%d\t%d\t%d\n"%(word2id[h], word2id[t], relation2id[r]))
	f.close()
	ff.close()

	f = open(export_path + "/entity2id.txt", "r")
	content = f.readlines()
	f.close()
	f = open(export_path + "/entity2id.txt", "w")
	f.write("%d\n"%(len(content)))
	for i in content:
		f.write(i.strip()+"\n")
	f.close()

	f = open(export_path + "/relation2id.txt", "r")
	content = f.readlines()
	f.close()
	f = open(export_path + "/relation2id.txt", "w")
	f.write("%d\n"%(len(content)))
	for i in content:
		f.write(i.strip()+"\n")
	f.close()

textual_rel_total, rel_total = init_relation()
entity_total, word_total = init_word()
desc,desc_total = init_entity_description(entity_total)

print (textual_rel_total)
print (rel_total)
print (entity_total)
print (word_total)
print (word_vec.shape)
print (desc_total)
f = open(data_path + "word2id.txt", "w")
for i in word2id:
	f.write("%s\t%d\n"%(i, word2id[i]))
f.close()

init_kg()
np.save(export_path+'vec', word_vec)
f = open(export_path+'config', "w")
f.write(json.dumps({"desc_len":desc_length,"desc_total":desc_total,"word2id":word2id,"relation2id":relation2id,"word_size":word_size, "fixlen":fixlen, "maxlen":maxlen, "entity_total":entity_total, "word_total":word_total, "rel_total":rel_total, "textual_rel_total":textual_rel_total}))
f.close()
sort_files("train", [textual_rel_total, rel_total])
sort_files("test", [textual_rel_total, rel_total])

# word_vec = np.load(export_path + 'vec.npy')
# f = open(export_path + "config", 'r')
# config = json.loads(f.read())
# f.close()
# relation2id = config["relation2id"]
# word2id = config["word2id"]

instance_triple, instance_scope, train_len, train_label, train_word, train_pos1, train_pos2, train_mask, train_head, train_tail,train_head_desc,train_tail_desc = init_train_files("train_sort",  [textual_rel_total, rel_total])
np.save(export_path+'train_instance_triple', instance_triple)## element (h,r,t),the sets of instance triple without repeat
np.save(export_path+'train_instance_scope', instance_scope) ## element[index1,index2] ,the index scope of the sentences instance that mention the same triple (h,r,t)
np.save(export_path+'train_len', train_len) ##[total_sentence] the length of sentence(<=fixlen)
np.save(export_path+'train_label', train_label) ## [total_sentence]relation that the sentence mentions
np.save(export_path+'train_word', train_word)  ##[total_sentence,fixlen]word id sequence of the sentence
np.save(export_path+'train_pos1', train_pos1)  ##[total_sentence]  position of the head entity of the sentence
np.save(export_path+'train_pos2', train_pos2)  ##[total_sentence] position of the tail entity of the sentence 
np.save(export_path+'train_mask', train_mask)  ##[total_sentence,fixlen]  ## 0,1,2,3 mask of the sentence after segmentation by head and tail entity
np.save(export_path+'train_head', train_head)  ## head entity of the sentence
np.save(export_path+'train_tail', train_tail)  ## tail entity of the sentence
np.save(export_path+'train_desc_head',train_head_desc)
np.save(export_path+'train_desc_tail',train_tail_desc)

instance_triple, instance_scope, test_len, test_label, test_word, test_pos1, test_pos2, test_mask, test_head, test_tail,test_desc_head,test_desc_tail = init_train_files("test_sort",  [textual_rel_total, rel_total])
np.save(export_path+'test_instance_triple', instance_triple)
np.save(export_path+'test_instance_scope', instance_scope)
np.save(export_path+'test_len', test_len)
np.save(export_path+'test_label', test_label)
np.save(export_path+'test_word', test_word)
np.save(export_path+'test_pos1', test_pos1)
np.save(export_path+'test_pos2', test_pos2)
np.save(export_path+'test_mask', test_mask)
np.save(export_path+'test_head', test_head)
np.save(export_path+'test_tail', test_tail)
np.save(export_path+'test_desc_head',test_desc_head)
np.save(export_path+'test_desc_tail',test_desc_tail)

np.save(export_path+'desc_all',desc)