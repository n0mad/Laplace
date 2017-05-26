import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class LaplacedModel:
    def __init__(self, batch_size, ewc):
        input_dim = 784
        output_dim = 10
        hidden_dim = 50

        self.batch_size = batch_size
        self.ewc = ewc

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, input_dim])
            self.y_ = tf.placeholder(tf.float32, [None, output_dim])

            self.W1 = tf.Variable(tf.zeros([input_dim, hidden_dim]), name='W1')
            self.b1 = tf.Variable(tf.zeros([hidden_dim]), name='b1')
            self.hidden = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)

            self.W2 = tf.Variable(tf.zeros([hidden_dim, output_dim]), name='W2')
            self.b2 = tf.Variable(tf.zeros([output_dim]), name='b2')

            self.y = tf.matmul(self.hidden, self.W2) + self.b2
            self.variables = [self.b1, self.W1, self.b2, self.W2]

            self.prev_solution_placeholders = [tf.placeholder(tf.float32, v.get_shape().as_list()) for v in self.variables]
            self.fisher_placeholders = [tf.placeholder(tf.float32, v.get_shape().as_list()) for v in self.variables]

            self.prediction_cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
            self.cost = self.prediction_cost
            for v, prev_v, fisher in zip(self.variables, self.prev_solution_placeholders, self.fisher_placeholders):
                self.cost = self.cost + tf.reduce_sum(tf.square(v - prev_v) * fisher) * 0.5

            is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


        self.tasks_learned = 0

        self.fisher_matrixes = [np.zeros(v.get_shape().as_list()) for v in self.variables]
        np.random.seed(13)
        self.prev_solution = [np.random.normal(size = v.get_shape().as_list(), scale = 0.01) for v in self.variables]

    def learn_new_task(self, data, validation_data):
        session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.train_step = tf.train.AdamOptimizer().minimize(self.cost)
            init = tf.global_variables_initializer()
            init.run(session=session)
            for variable, value in zip(self.variables, self.prev_solution):
                session.run(variable.assign(value))

        average = 0.0
        feed_dict = dict( zip(self.fisher_placeholders, self.fisher_matrixes) + zip(self.prev_solution_placeholders, self.prev_solution) )

        for i in xrange(1, 5000):
            batch = data.next_batch(self.batch_size)
            feed_dict[self.x] = batch[0]
            feed_dict[self.y_] = batch[1]
            _, batch_accuracy = session.run([self.train_step, self.accuracy], feed_dict=feed_dict)
            average += batch_accuracy

            #if i % 100 == 0:
            #    print average / 100
            #    average = 0.0
        self.tasks_learned += 1
        if self.ewc:
            self.update_fisher_diag(session, data)
            self.prev_solution = session.run(self.variables)
            #for i, v in enumerate(self.variables):
            #    print v.name, np.linalg.norm(self.fisher_matrixes[i]), np.linalg.norm(self.prev_solution[i])

        x, y_ = validation_data.images, validation_data.labels
        accuracy = session.run([self.accuracy], feed_dict={self.x: x, self.y_: y_})
        return accuracy

    def update_fisher_diag(self, session, dataset):
        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
	for i in xrange(100):
            im_ind = np.random.randint(dataset.images.shape[0])
            ders = session.run(tf.gradients(tf.log(probs[0,class_ind]), self.variables), feed_dict={self.x: dataset.images[im_ind:im_ind+1]})
            for f, d in zip(self.fisher_matrixes, ders):
                f += d * d / 100

def split_dataset(dataset, parts):
    n = dataset._num_examples
    perm = np.arange(n)
    np.random.shuffle(perm)

    datasets = []
    images = dataset.images[perm]
    labels = dataset.labels[perm]

    for part in xrange(parts):
        start = n / parts * part
        end = min(n, start + n / parts)

        datasets.append(dataset.__class__(images[start:end], labels[start:end], reshape=False))

    return datasets

def get_learning_curve(model, train, test, sequential_tasks):
    np.random.seed(13)
    sub_problems = split_dataset(train, sequential_tasks)

    loss_values = []
    for task_id, task in enumerate(sub_problems):
        print 'learning task #', task_id
        validation_loss = model.learn_new_task(task, test)
        print 'validation loss:', validation_loss
        loss_values.append(validation_loss)
    return loss_values

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sequential_tasks = 5

    model_laplaced = LaplacedModel(8, True)
    laplaced_loss_curve = get_learning_curve(model_laplaced, mnist.train, mnist.test, sequential_tasks)

    model_naive = LaplacedModel(8, False)
    naive_loss_curve = get_learning_curve(model_naive, mnist.train, mnist.test, sequential_tasks)

    model_baseline = LaplacedModel(8, False)
    naive_loss_curve = get_learning_curve(model_baseline, mnist.train, mnist.test, sequential_tasks)

