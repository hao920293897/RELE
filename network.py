import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope as vs

FLAGS = tf.app.flags.FLAGS


class NN(object):
    def __init__(self, is_training, word_embeddings,ent_embedding,rel_embedding, simple_position=False):
        self.max_length = FLAGS.max_length
        self.num_classes = FLAGS.num_classes
        self.word_size = len(word_embeddings[0])
        self.hidden_size = FLAGS.hidden_size
        self.description_length = FLAGS.description_length
        self.alpha = FLAGS.alpha
        self.beta = 0.1
        self.weight_decay = FLAGS.weight_decay
        self.DESC = FLAGS.desc_flag
        self.gate = FLAGS.gate
        if FLAGS.model.lower() == "cnn":
            self.output_size = FLAGS.hidden_size
        elif FLAGS.model.lower() == "pcnn":
            self.output_size = FLAGS.hidden_size * 3
        elif FLAGS.model.lower() == "lstm":
            self.output_size = FLAGS.hidden_size
        elif FLAGS.model.lower() == "gru":
            self.output_size = FLAGS.hidden_size
        elif FLAGS.model.lower() == "bi-lstm" or FLAGS.model.lower() == "bilstm":
            self.output_size = FLAGS.hidden_size * 2
        elif FLAGS.model.lower() == "bi-gru" or FLAGS.model.lower() == "bigru":
            self.output_size = FLAGS.hidden_size * 2
        self.margin = FLAGS.margin
        # placeholders for text models
        self.word = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='input_word')
        self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='input_pos1')
        self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='input_pos2')
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='input_mask')
        self.len = tf.placeholder(dtype=tf.int32, shape=[None], name='input_len')
        self.label_index = tf.placeholder(dtype=tf.int32, shape=[None], name='label_index')
        self.head_index = tf.placeholder(dtype=tf.int32, shape=[None], name='head_index')
        self.tail_index = tf.placeholder(dtype=tf.int32, shape=[None], name='tail_index')
        self.label = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, self.num_classes], name='input_label')
        self.scope = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size + 1], name='scope')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.weights = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size])
        # descriptions
        self.head_description = tf.placeholder(dtype=tf.int32, shape=[None, self.description_length],
                                               name='head_description')
        self.tail_description = tf.placeholder(dtype=tf.int32, shape=[None, self.description_length],
                                               name='tail_description')

        with tf.name_scope("embedding-layers"):
            if self.gate:
                self.gate_theta0 = tf.get_variable(name='gate_theta',shape=[FLAGS.ent_total,self.word_size],
                                                  initializer=tf.contrib.layers.xavier_initializer())
                self.gate_theta = tf.nn.sigmoid(self.gate_theta0)
            # word embeddings
            temp_word_embedding = tf.get_variable(initializer=word_embeddings[FLAGS.ent_total:, :],
                                                  name='temp_word_embedding', dtype=tf.float32)
            self.ent_embedding = tf.get_variable(name="ent_embedding",shape = [FLAGS.ent_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            # self.ent_embedding = tf.get_variable(name="ent_embedding", initializer=ent_embedding)
            unk_word_embedding = tf.get_variable('unk_embedding', [self.word_size], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())

            self.word_embedding = tf.concat([
                self.ent_embedding,
                temp_word_embedding,
                tf.reshape(unk_word_embedding, [1, self.word_size]),
                tf.reshape(tf.constant(np.zeros(self.word_size, dtype=np.float32)), [1, self.word_size])], 0)
            # self.word_embedding = tf.nn.l2_normalize(self.word_embedding)

            if self.DESC:
                self.relation_matrix = tf.get_variable('relation_matrix',
                                                        [self.num_classes, 2*self.word_size+self.output_size],
                                                        dtype=tf.float32,
                                                        initializer=tf.contrib.layers.xavier_initializer())
            else:
                self.relation_matrix = tf.get_variable('relation_matrix',
                                                       [self.num_classes, self.output_size],
                                                       dtype=tf.float32,
                                                       initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable('bias', [self.num_classes], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.latent_bias = tf.get_variable('latent_bias', [self.num_classes], dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer())
            self.latent_matrix = tf.get_variable('latent_matrix',
                                                 [self.num_classes, self.word_size],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            # position embeddings
            if simple_position:
                temp_pos_array = np.zeros((FLAGS.pos_num + 1, FLAGS.pos_size), dtype=np.float32)
                temp_pos_array[(FLAGS.pos_num - 1) / 2] = np.ones(FLAGS.pos_size, dtype=np.float32)
                self.pos1_embedding = tf.constant(temp_pos_array)
                self.pos2_embedding = tf.constant(temp_pos_array)
            else:
                temp_pos1_embedding = tf.get_variable('temp_pos1_embedding', [FLAGS.pos_num, FLAGS.pos_size],
                                                      dtype=tf.float32,
                                                      initializer=tf.contrib.layers.xavier_initializer())
                temp_pos2_embedding = tf.get_variable('temp_pos2_embedding', [FLAGS.pos_num, FLAGS.pos_size],
                                                      dtype=tf.float32,
                                                      initializer=tf.contrib.layers.xavier_initializer())
                self.pos1_embedding = tf.concat([temp_pos1_embedding,
                                                 tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)),
                                                            [1, FLAGS.pos_size])], 0)
                self.pos2_embedding = tf.concat([temp_pos2_embedding,
                                                 tf.reshape(tf.constant(np.zeros(FLAGS.pos_size, dtype=np.float32)),
                                                            [1, FLAGS.pos_size])], 0)
            # relation embeddings and the transfer matrix between relations and textual relations
            # self.rel_embeddings = tf.get_variable(name="rel_embedding",initializer=rel_embedding)
            # self.rel_embeddings = tf.nn.l2_normalize(self.DSREAD)
            # self.rel_embeddings = tf.get_variable(name="rel_embedding", shape = [FLAGS.rel_total, self.word_size], initializer = tf.contrib.layers.xavier_initializer(uniform=False))
            self.transfer_matrix = tf.get_variable("transfer_matrix", [self.output_size+2*self.word_size, self.word_size])
            self.transfer_matrix_desc = tf.get_variable("transfer_matrix_desc",[self.word_size,self.word_size])
            # self.transfer_matrix_word = tf.get_variable("transfer_matrix_word", [self.output_size, self.word_size])
            self.transfer_bias = tf.get_variable('transfer_bias', [self.word_size], dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.transfer_bias_desc = tf.get_variable("transfer_bias_desc",[self.word_size],dtype=tf.float32,
                                                      initializer=tf.contrib.layers.xavier_initializer())
            # self.transfer_matrix_mention = tf.get_variable("transfer_matrix_mention",[self.word_size*2+self.output_size,self.word_size])
            # self.transfer_bias_mention = tf.get_variable('transfer_bias_mention', [self.word_size], dtype=tf.float32,
            #                                      initializer=tf.contrib.layers.xavier_initializer())
            # self.transfer_matrix_d = tf.get_variable("transfer_matrix_d", [self.output_size, self.word_size])
            # # self.transfer_matrix_word = tf.get_variable("transfer_matrix_word", [self.output_size, self.word_size])
            # self.transfer_bias_d = tf.get_variable('transfer_bias_d', [self.word_size], dtype=tf.float32,
            #                                      initializer=tf.contrib.layers.xavier_initializer())

        with tf.name_scope("embedding-lookup"):
            # textual embedding-lookup
            input_word = tf.nn.embedding_lookup(self.word_embedding, self.word)
            input_pos1 = tf.nn.embedding_lookup(self.pos1_embedding, self.pos1)
            input_pos2 = tf.nn.embedding_lookup(self.pos2_embedding, self.pos2)
            self.input_embedding = tf.concat(values=[input_word, input_pos1, input_pos2], axis=2)
            self.head_desc_embedding = tf.nn.embedding_lookup(self.word_embedding, self.head_description)
            self.tail_desc_embedding = tf.nn.embedding_lookup(self.word_embedding, self.tail_description)

    def transfer(self, x):
        res = tf.tanh(tf.nn.bias_add(tf.matmul(x, self.transfer_matrix), self.transfer_bias))
        return res
    def transfer_mention(self, x):
        res = tf.tanh(tf.nn.bias_add(tf.matmul(x, self.transfer_matrix_mention), self.transfer_bias_mention))
        return res

    def desc_transfer(self, desc_embedding):
        # res = tf.tanh(
        #     tf.nn.bias_add(self.multiply_tensors(desc_embedding, self.transfer_matrix_desc), self.transfer_bias_desc))
        desc_len = int(desc_embedding.shape[1])
        v = tf.matmul(tf.reshape(desc_embedding,[-1,self.word_size]),self.transfer_matrix_desc)
        vu = tf.reshape(v,[-1,desc_len,self.word_size])
        res = tf.tanh(tf.nn.bias_add(vu,self.transfer_bias_desc))
        return res

    # def mention_transfer(self, mention_embedding):
    #     # res = tf.tanh(tf.nn.bias_add(self.multiply_tensors(mention_embedding,tf.transpose(self.transfer_matrix_mention)),self.transfer_bias_mention))
    #     # size = mention_embedding.shape[0].value
    #     # weights = tf.expand_dims(self.transfer_matrix_mention, 0)
    #     # weights = tf.tile(weights, [size, 1, 1])
    #     # res = tf.tanh(tf.nn.bias_add(mention_embedding * weights, self.transfer_bias_mention))
    #     desc_len = int(mention_embedding.shape[1])
    #     v = tf.matmul(tf.reshape(mention_embedding, [-1, self.word_size*2]), self.transfer_matrix_mention)
    #     vu = tf.reshape(v, [-1, desc_len, self.output_size])
    #
    #     res = tf.tanh(tf.nn.bias_add(vu, self.transfer_bias_mention))
    #     return res

    def set_alpha(self,desc):
        desc_sum = tf.reduce_sum(desc,1)
        zero_idx = tf.squeeze(tf.where(desc_sum>0))
        alpha = self.alpha*tf.ones(tf.shape(zero_idx)[0])
        weights = tf.SparseTensor(indices=tf.squeeze(zero_idx),values=alpha,dense_shape=[tf.cast(tf.shape(desc)[0],dtype=tf.int64)])
        # alpha = tf.reshape(alpha,[-1,1])
        return tf.reshape(tf.sparse_tensor_to_dense(weights),[-1,1])


    def att(self, x, is_training=True, dropout=True):
        with tf.name_scope("sentence-level-attention"):
            current_attention = tf.nn.embedding_lookup(self.relation_matrix, self.label_index)
            attention_logit = tf.reduce_sum(current_attention * x, 1)
            tower_repre = []
            for i in range(FLAGS.batch_size):
                sen_matrix = x[self.scope[i]:self.scope[i + 1]]
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i + 1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix), [self.output_size])
                tower_repre.append(final_repre)
            if dropout:
                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate=self.keep_prob, training=is_training)
            else:
                stack_repre = tf.stack(tower_repre)
        return stack_repre

    def katt(self, x,head_desc,tail_desc,is_training=True, dropout=True):
        with tf.name_scope("knowledge-based-attention"):
            head = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
            tail = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)

            relation = tail - head
            if FLAGS.desc_att:
                head_desc_1 = self.desc_kg_att(desc_embedding=head_desc, kg_embedding=relation)
                tail_desc_1 = self.desc_kg_att(desc_embedding=tail_desc, kg_embedding=relation)
            else:
                head_desc_1 = head_desc
                tail_desc_1 = tail_desc
            gate_l2 = 0
            if self.DESC:
                if self.gate:
                    h_gate = tf.nn.embedding_lookup(self.gate_theta,self.head_index)
                    t_gate = tf.nn.embedding_lookup(self.gate_theta,self.tail_index)
                    head_repre = h_gate * head + (1 - h_gate) * head_desc_1
                    tail_repre = t_gate * tail + (1 - t_gate) * tail_desc_1
                    gate_l2 += tf.nn.l2_loss(h_gate)
                    gate_l2 += tf.nn.l2_loss(t_gate)
                else:
                    head_alpha = self.set_alpha(self.head_description)
                    tail_alpha = self.set_alpha(self.tail_description)
                    head_repre = (1-head_alpha) * head + head_alpha * head_desc_1
                    tail_repre = (1-tail_alpha) * tail + tail_alpha * tail_desc_1
                kg_att = tail_repre - head_repre
            else:
                head_repre = head
                tail_repre = tail
                kg_att = relation
            # head_mention_embd = self.desc_mention_att(desc_embedding=head_desc,mention_embedding=x)
            # tail_mention_embd = self.desc_mention_att(desc_embedding=tail_desc, mention_embedding=x)
            # desc_embedding = tf.nn.bias_add(tf.matmul(tf.concat([head_mention_embd,tail_mention_embd],1),self.transfer_matrix_d),self.transfer_bias_d)
            desc_embedding = tf.nn.l2_normalize(tf.concat([head_desc_1,tail_desc_1],1))

            # desc_embedding = self.desc_mention_att(desc_embedding=tf.concat([head_desc,tail_desc],2),
            #                                        mention_embedding=x)
            mention_embedding = tf.concat([x,desc_embedding],1)
            attention_logit = tf.reduce_sum( self.transfer(mention_embedding) * kg_att, 1)
            tower_repre = []

            head_repre_g = []
            tail_repre_g = []
            for i in range(FLAGS.batch_size):
                sen_matrix = mention_embedding[self.scope[i]:self.scope[i + 1]] ##get the sentence bag that contains the same entity
                attention_score = tf.nn.softmax(tf.reshape(attention_logit[self.scope[i]:self.scope[i + 1]], [1, -1]))
                final_repre = tf.reshape(tf.matmul(attention_score, sen_matrix), [self.word_size*2+self.output_size])
                tower_repre.append(final_repre)
                head_repre_g.append(head_repre[self.scope[i]])
                tail_repre_g.append(tail_repre[self.scope[i]])
            if dropout:
                stack_repre = tf.layers.dropout(tf.stack(tower_repre), rate=self.keep_prob, training=is_training)
            else:
                stack_repre = tf.stack(tower_repre)
            head_repre_g = tf.stack(head_repre_g)
            tail_repre_g = tf.stack(tail_repre_g)
        return stack_repre,head_repre_g,tail_repre_g

    def desc_kg_att(self,desc_embedding,kg_embedding,is_training=True,dropout=True):
        with tf.name_scope('desc_kg_att'):
            x = self.desc_transfer(desc_embedding)
            kg_embedding = tf.expand_dims(kg_embedding,1)
            attention_logit = tf.reduce_sum(x*kg_embedding,2)
            # attention_logit = tf.reduce_sum(x*kg_embedding)
            attention_score = tf.nn.softmax(attention_logit)
            attention_score = tf.expand_dims(attention_score,2)
            entity_repre = tf.reduce_sum(attention_score*x,1)
            if dropout:
                stack_repre = tf.layers.dropout(entity_repre, rate=self.keep_prob, training=is_training)
            else:
                stack_repre = entity_repre
            return  stack_repre

    def desc_mention_att(self,desc_embedding,mention_embedding,is_training=True,dropout=True):
        with tf.name_scope('desc_men_att'):
            desc_embedding_trans = desc_embedding
            men_embedding = tf.expand_dims((mention_embedding),1)
            attention_logit = tf.reduce_sum(desc_embedding_trans*men_embedding,2)
            attention_score = tf.nn.softmax(attention_logit)
            attention_score = tf.expand_dims(attention_score,2)
            rel_repre = tf.reduce_sum(attention_score*desc_embedding,1)
            # print('rel_repre:', rel_repre.shape)
            # rel_repre = tf.tanh(tf.nn.bias_add(tf.matmul(rel_repre, self.transfer_matrix_desc), self.transfer_bias_desc))
            if dropout:
                stack_repre = tf.layers.dropout(rel_repre, rate=self.keep_prob, training=is_training)
            else:
                stack_repre = rel_repre
            return  stack_repre
    def desc_cnn(self, desc_input):
        with tf.name_scope('description-cnn'):
            with tf.variable_scope('desc_cnn',reuse=tf.AUTO_REUSE) as scope:
                input_description = tf.expand_dims(desc_input, axis=1)
                x = tf.layers.conv2d(inputs=input_description, filters=self.word_size, kernel_size=[1, 3], strides=[1, 1],
                                     padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                # x = tf.reduce_max(x, axis=2)
                # x = tf.squeeze(x,[1])
                if FLAGS.desc_att:
                    x = tf.nn.relu(tf.squeeze(x,[1]))
                else:
                    x = tf.reduce_max(x,axis=2)
                    x = tf.squeeze(x,[1])
        return x

    def att_test(self, x, is_training=False):
        test_attention_logit = tf.matmul(x, tf.transpose(self.relation_matrix))
        return test_attention_logit

    def katt_test(self, x,head_desc,tail_desc,is_training=False):
        head = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
        tail = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)

        # each_att = tf.expand_dims(tail - head, -1)
        relation = tail - head
        if FLAGS.desc_att:
            tail_desc_1,head_desc_att = self.desc_kg_att_test(desc_embedding=tail_desc, kg_embedding=relation)
            head_desc_1,tail_desc_att = self.desc_kg_att_test(desc_embedding=head_desc, kg_embedding=relation)
        else :
            head_desc_1 = head_desc
            tail_desc_1 = tail_desc
            head_desc_att = 0
            tail_desc_att = 0
        # cx = tf.concat([x,head_desc_1,tail_desc_1],1)
        if self.DESC:
            if self.gate:
                h_gate = tf.nn.embedding_lookup(self.gate_theta,self.head_index)
                t_gate = tf.nn.embedding_lookup(self.gate_theta,self.tail_index)
                head_repre = h_gate * head + (1 - h_gate) * head_desc_1
                tail_repre = t_gate * tail + (1 - t_gate) * tail_desc_1
            else:
                head_alpha = self.set_alpha(self.head_description)
                tail_alpha = self.set_alpha(self.tail_description)
                head_repre = (1-head_alpha) * head + head_alpha * head_desc_1
                tail_repre = (1-tail_alpha) * tail + tail_alpha * tail_desc_1
            each_att = tf.expand_dims((tail_repre - head_repre),-1)
        else:
            head_repre = head
            tail_repre = tail
            each_att = tf.expand_dims(relation, -1)
        kg_att = tf.concat([each_att for i in range(self.num_classes)], 2)

        # head_mention_embd = self.desc_mention_att(desc_embedding=head_desc, mention_embedding=x,dropout=False,is_training=False)
        # tail_mention_embd = self.desc_mention_att(desc_embedding=tail_desc, mention_embedding=x,dropout=False,is_training=False)
        # desc_embedding = tf.nn.bias_add(tf.matmul(tf.concat([head_mention_embd, tail_mention_embd], 1), self.transfer_matrix_d),
        #                    self.transfer_bias_d)
        # desc_embedding = self.desc_mention_att(desc_embedding=tf.concat([head_desc,tail_desc],2),mention_embedding=x,dropout=False,is_training=False)
        desc_embedding = tf.nn.l2_normalize(tf.concat([head_desc_1,tail_desc_1],1))
        mention_embedding = tf.concat([x, desc_embedding], 1)
        x = tf.reshape(self.transfer(mention_embedding), [-1, 1, self.word_size])
        test_attention_logit = tf.matmul(x, kg_att)
        return tf.reshape(test_attention_logit, [-1, self.num_classes]),head_repre,tail_repre,mention_embedding,head_desc_att,tail_desc_att

    def desc_kg_att_test(self, desc_embedding, kg_embedding):
        # x = self.desc_cnn(desc_embedding)
        x = self.desc_transfer(desc_embedding)
        kg_embedding = tf.expand_dims(kg_embedding, 1)
        attention_logit = tf.reduce_sum(x * kg_embedding, 2)
        # attention_logit = tf.reduce_sum(x*kg_embedding)
        attention_score = tf.nn.softmax(attention_logit)
        attention_score = tf.expand_dims(attention_score, 2)
        entity_repre = tf.reduce_sum(attention_score * x, 1)

        return entity_repre,attention_score

class CNN(NN):
    def __init__(self, is_training, word_embeddings,ent_embedding,rel_embedding,simple_position=False):
        NN.__init__(self, is_training, word_embeddings,ent_embedding,rel_embedding, simple_position)

        with tf.name_scope("conv-maxpool"):

            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            x = tf.layers.conv2d(inputs=input_sentence, filters=FLAGS.hidden_size, kernel_size=[1, 3], strides=[1, 1],
                                 padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            x = tf.reduce_max(x, axis=2)
            x = tf.nn.relu(tf.squeeze(x,[1]))

            # head = tf.nn.embedding_lookup(self.word_embedding, self.head_index)
            # tail = tf.nn.embedding_lookup(self.word_embedding, self.tail_index)
            # relation = tail-head
            #
            # if is_training:
            #     x = self.word_kg_att(mention_embedding=x,kg_embedding=relation,dropout=False,is_training=False)
            # else:
            #     x = self.word_kg_att(mention_embedding=x,kg_embedding=relation,dropout=False,is_training=False)

        with tf.name_scope("DSREAD-desc_cnn"):
            head_desc = self.desc_cnn(self.head_desc_embedding)
            tail_desc = self.desc_cnn(self.tail_desc_embedding)
            self.l2_loss_nn = 0

        if FLAGS.katt_flag != 0:
            gate_l2 = 0
            stack_repre,head_repre,tail_repre = self.katt(x,head_desc,tail_desc, is_training)
            self.l2_loss_nn = gate_l2

        else:
            stack_repre = self.att(x, is_training)

        with tf.name_scope("loss"):
            logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
            if self.DESC:
                latent_repre = tail_repre-head_repre
                latent_logits = tf.matmul(latent_repre,tf.transpose(self.latent_matrix))+self.latent_bias
                with tf.name_scope("l2_regularization"):
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.gate_theta)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.gate_theta)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_matrix_desc)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_bias_desc)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_matrix)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_bias)
                    # tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.latent_matrix)
                    # tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.latent_bias)
                    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
                    reg_term = tf.contrib.layers.apply_regularization(regularizer)
                    self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits)) + tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                                logits=latent_logits)) + reg_term
            else:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))

            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss', self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            with tf.name_scope("test"):
                if FLAGS.katt_flag != 0:
                    test_attention_logit,head_repre,tail_repre,t,self.head_desc_att,self.tail_desc_att = self.katt_test(x,head_desc,tail_desc)

                else:
                    test_attention_logit = self.att_test(x)
                    t = x

                test_tower_output = []
                atttention_score = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(tf.transpose(test_attention_logit[self.scope[i]:self.scope[i + 1], :]))
                    # print(test_attention_score)
                    atttention_score.append(test_attention_score)
                    final_repre = tf.matmul(test_attention_score, t[self.scope[i]:self.scope[i + 1]])
                    logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output), [FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output
                # self.attention_score = tf.stack(atttention_score)

class PCNN(NN):
    def __init__(self, is_training, word_embeddings,ent_embedding, rel_embedding, simple_position=False):
        NN.__init__(self, is_training, word_embeddings, ent_embedding, rel_embedding, simple_position)
        with tf.name_scope("conv-maxpool"):
            mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            pcnn_mask = tf.nn.embedding_lookup(mask_embedding, self.mask)
            input_sentence = tf.expand_dims(self.input_embedding, axis=1)
            x = tf.layers.conv2d(inputs=input_sentence, filters=FLAGS.hidden_size, kernel_size=[1, 3], strides=[1, 1],
                                 padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            x = tf.reshape(x, [-1, self.max_length, FLAGS.hidden_size, 1])
            x = tf.reduce_max(tf.reshape(pcnn_mask, [-1, 1, self.max_length, 3]) * tf.transpose(x, [0, 2, 1, 3]),
                              axis=2)
            x = tf.nn.relu(tf.reshape(x, [-1, self.output_size]))
            # print('x:', x.get_shape().as_list())

        with tf.name_scope("DSREAD-desc_cnn"):
            head_desc = self.desc_cnn(self.head_desc_embedding)
            tail_desc = self.desc_cnn(self.tail_desc_embedding)
            self.l2_loss_nn = 0

        if FLAGS.katt_flag != 0:
            gate_l2 = 0
            stack_repre, head_repre, tail_repre = self.katt(x, head_desc, tail_desc, is_training)
            self.l2_loss_nn = gate_l2
        else:
            stack_repre = self.att(x, is_training)

        with tf.name_scope("loss"):
            logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=logits, weights=self.weights)
            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss', self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

            with tf.name_scope("loss"):
                latent_repre = tail_repre - head_repre
                latent_logits = tf.matmul(latent_repre, tf.transpose(self.latent_matrix)) + self.latent_bias
                logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias

                # self.l2_loss_nn += tf.nn.l2_loss(self.transfer_matrix_desc)
                # self.l2_loss_nn += tf.nn.l2_loss(self.transfer_matrix)
                # self.l2_loss_nn += tf.nn.l2_loss(self.transfer_bias)
                # self.l2_loss_nn += tf.nn.l2_loss(self.transfer_bias_desc)
                # self.l2_loss_nn += tf.nn.l2_loss(self.latent_bias)
                # self.l2_loss_nn += tf.nn.l2_loss(self.latent_matrix)
                with tf.name_scope("l2_regularization"):
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.gate_theta)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.gate_theta)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_matrix_desc)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_bias_desc)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_matrix)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.transfer_bias)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.latent_matrix)
                    tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.latent_bias)
                    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/50000)
                    reg_term = tf.contrib.layers.apply_regularization(regularizer)

                # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits)) + tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                            logits=latent_logits))+reg_term

                self.output = tf.nn.softmax(logits)
                tf.summary.scalar('loss', self.loss)
                self.predictions = tf.argmax(logits, 1, name="predictions")
                self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

            if not is_training:
                with tf.name_scope("test"):
                    if FLAGS.katt_flag != 0:
                        test_attention_logit, head_repre, tail_repre, t, self.head_desc_att, self.tail_desc_att = self.katt_test(
                            x, head_desc, tail_desc)

                    else:
                        test_attention_logit = self.att_test(x)

                    test_tower_output = []
                    atttention_score = []
                    for i in range(FLAGS.test_batch_size):
                        test_attention_score = tf.nn.softmax(
                            tf.transpose(test_attention_logit[self.scope[i]:self.scope[i + 1], :]))
                        # print(test_attention_score)
                        atttention_score.append(test_attention_score)
                        final_repre = tf.matmul(test_attention_score, t[self.scope[i]:self.scope[i + 1]])
                        logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                        output = tf.diag_part(tf.nn.softmax(logits))
                        test_tower_output.append(output)
                    test_stack_output = tf.reshape(tf.stack(test_tower_output),
                                                   [FLAGS.test_batch_size, self.num_classes])
                    self.test_output = test_stack_output
                    # self.attention_score = tf.stack(atttention_score)
        #
        # if not is_training:
        #     with tf.name_scope("test"):
        #         if FLAGS.katt_flag != 0:
        #             test_attention_logit,head_repre,tail_repre = self.katt_test(x,head_desc,tail_desc)
        #         else:
        #             test_attention_logit = self.att_test(x)
        #         test_tower_output = []
        #         for i in range(FLAGS.test_batch_size):
        #             test_attention_score = tf.nn.softmax(
        #                 tf.transpose(test_attention_logit[self.scope[i]:self.scope[i + 1], :]))
        #             final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i + 1]])
        #             logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
        #             output = tf.diag_part(tf.nn.softmax(logits))
        #             test_tower_output.append(output)
        #         test_stack_output = tf.reshape(tf.stack(test_tower_output), [FLAGS.test_batch_size, self.num_classes])
        #         self.test_output = test_stack_output


class RNN(NN):
    def get_rnn_cell(self, dim, cell_name='lstm'):
        if isinstance(cell_name, list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return self.get_rnn_cell(dim, cell_name[0])
            cells = [self.get_rnn_cell(dim, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(dim)
        raise NotImplementedError

    def __init__(self, is_training, word_embeddings, cell_name, simple_position=False):
        NN.__init__(self, is_training, word_embeddings, simple_position)
        input_sentence = tf.layers.dropout(self.input_embedding, rate=self.keep_prob, training=is_training)
        with tf.name_scope('rnn'):
            cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
            outputs, states = tf.nn.dynamic_rnn(cell, input_sentence,
                                                sequence_length=self.len,
                                                dtype=tf.float32,
                                                scope='dynamic-rnn')
            if isinstance(states, tuple):
                states = states[0]
            x = states

        if FLAGS.katt_flag != 0:
            stack_repre = self.katt(x, is_training, False)
        else:
            stack_repre = self.att(x, is_training, False)

        with tf.name_scope("loss"):
            logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=logits, weights=self.weights)
            self.output = tf.nn.softmax(logits)
            tf.summary.scalar('loss', self.loss)
            self.predictions = tf.argmax(logits, 1, name="predictions")
            self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

        if not is_training:
            with tf.name_scope("test"):
                if FLAGS.katt_flag != 0:
                    test_attention_logit = self.katt_test(x)
                else:
                    test_attention_logit = self.att_test(x)
                test_tower_output = []
                for i in range(FLAGS.test_batch_size):
                    test_attention_score = tf.nn.softmax(
                        tf.transpose(test_attention_logit[self.scope[i]:self.scope[i + 1], :]))
                    final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i + 1]])
                    logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                    output = tf.diag_part(tf.nn.softmax(logits))
                    test_tower_output.append(output)
                test_stack_output = tf.reshape(tf.stack(test_tower_output), [FLAGS.test_batch_size, self.num_classes])
                self.test_output = test_stack_output


class BiRNN(NN):
    def get_rnn_cell(self, dim, cell_name='lstm'):
        if isinstance(cell_name, list) or isinstance(cell_name, tuple):
            if len(cell_name) == 1:
                return self.get_rnn_cell(dim, cell_name[0])
            cells = [self.get_rnn_cell(dim, c) for c in cell_name]
            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        if cell_name.lower() == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)
        elif cell_name.lower() == 'gru':
            return tf.contrib.rnn.GRUCell(dim)
        raise NotImplementedError

    def __init__(self, is_training, word_embeddings, cell_name, simple_position=False):
        NN.__init__(self, is_training, word_embeddings, simple_position)
        input_sentence = tf.layers.dropout(self.input_embedding, rate=self.keep_prob, training=is_training)
        with tf.name_scope('bi-rnn'):
            fw_cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
            bw_cell = self.get_rnn_cell(FLAGS.hidden_size, cell_name)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, input_sentence,
                sequence_length=self.len,
                dtype=tf.float32,
                scope='bi-dynamic-rnn')
            fw_states, bw_states = states
            if isinstance(fw_states, tuple):
                fw_states = fw_states[0]
                bw_states = bw_states[0]
            x = tf.concat(states, axis=1)

            with tf.name_scope("DSREAD-desc_cnn"):
                head_desc = self.desc_cnn(self.head_desc_embedding)
                tail_desc = self.desc_cnn(self.tail_desc_embedding)
                self.l2_loss_nn = 0

            if FLAGS.katt_flag != 0:
                gate_l2 = 0
                stack_repre, head_repre, tail_repre = self.katt(x, head_desc, tail_desc, is_training)
                self.l2_loss_nn = gate_l2
            else:
                stack_repre = self.att(x, is_training)

            with tf.name_scope("loss"):
                logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=logits,
                                                            weights=self.weights)
                self.output = tf.nn.softmax(logits)
                tf.summary.scalar('loss', self.loss)
                self.predictions = tf.argmax(logits, 1, name="predictions")
                self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

                with tf.name_scope("loss"):
                    latent_repre = tail_repre - head_repre
                    latent_logits = tf.matmul(latent_repre, tf.transpose(self.latent_matrix)) + self.latent_bias
                    logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
                    self.l2_loss_nn += tf.nn.l2_loss(self.transfer_matrix_desc)
                    self.l2_loss_nn += tf.nn.l2_loss(self.transfer_matrix)
                    self.l2_loss_nn += tf.nn.l2_loss(self.transfer_bias)
                    self.l2_loss_nn += tf.nn.l2_loss(self.transfer_bias_desc)
                    self.l2_loss_nn += tf.nn.l2_loss(self.latent_bias)
                    self.l2_loss_nn += tf.nn.l2_loss(self.latent_matrix)
                    # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
                    self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits)) + tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                                logits=latent_logits)) + self.weight_decay * self.l2_loss_nn
                    self.output = tf.nn.softmax(logits)
                    tf.summary.scalar('loss', self.loss)
                    self.predictions = tf.argmax(logits, 1, name="predictions")
                    self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

                if not is_training:
                    with tf.name_scope("test"):
                        if FLAGS.katt_flag != 0:
                            test_attention_logit, head_repre, tail_repre, t, self.head_desc_att, self.tail_desc_att = self.katt_test(
                                x, head_desc, tail_desc)

                        else:
                            test_attention_logit = self.att_test(x)

                        test_tower_output = []
                        atttention_score = []
                        for i in range(FLAGS.test_batch_size):
                            test_attention_score = tf.nn.softmax(
                                tf.transpose(test_attention_logit[self.scope[i]:self.scope[i + 1], :]))
                            # print(test_attention_score)
                            atttention_score.append(test_attention_score)
                            final_repre = tf.matmul(test_attention_score, t[self.scope[i]:self.scope[i + 1]])
                            logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
                            output = tf.diag_part(tf.nn.softmax(logits))
                            test_tower_output.append(output)
                        test_stack_output = tf.reshape(tf.stack(test_tower_output),
                                                       [FLAGS.test_batch_size, self.num_classes])
                        self.test_output = test_stack_output
        # if FLAGS.katt_flag != 0:
        #     stack_repre = self.katt(x, is_training, False)
        # else:
        #     stack_repre = self.att(x, is_training, False)
        #
        # with tf.name_scope("loss"):
        #     logits = tf.matmul(stack_repre, tf.transpose(self.relation_matrix)) + self.bias
        #     self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=logits))
        #     self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=logits, weights=self.weights)
        #     self.output = tf.nn.softmax(logits)
        #     tf.summary.scalar('loss', self.loss)
        #     self.predictions = tf.argmax(logits, 1, name="predictions")
        #     self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        #
        # if not is_training:
        #     with tf.name_scope("test"):
        #         if FLAGS.katt_flag != 0:
        #             test_attention_logit = self.katt_test(x)
        #         else:
        #             test_attention_logit = self.att_test(x)
        #         test_tower_output = []
        #         for i in range(FLAGS.test_batch_size):
        #             test_attention_score = tf.nn.softmax(
        #                 tf.transpose(test_attention_logit[self.scope[i]:self.scope[i + 1], :]))
        #             final_repre = tf.matmul(test_attention_score, x[self.scope[i]:self.scope[i + 1]])
        #             logits = tf.matmul(final_repre, tf.transpose(self.relation_matrix)) + self.bias
        #             output = tf.diag_part(tf.nn.softmax(logits))
        #             test_tower_output.append(output)
        #         test_stack_output = tf.reshape(tf.stack(test_tower_output), [FLAGS.test_batch_size, self.num_classes])
        #         self.test_output = test_stack_output
