import tensorflow as tf

class Model:

    def __init__(self, n, d, seed=8, MLEScore='Sup-G', l1_lambda=0.1, use_float64=False):

        self.n = n
        self.d = d
        self.seed = seed
        self.MLEScore = MLEScore
        self.l1_lambda = l1_lambda
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32

        # Initializer (for reproducibility)
        self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)

        self._build()
        self._init_session()

    def _init_session(self):
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=0.5,
                allow_growth=True,
            )
        ))

    def _build(self):
        tf.compat.v1.reset_default_graph()

        self.rho = tf.compat.v1.placeholder(self.tf_float_type)
        self.alpha = tf.compat.v1.placeholder(self.tf_float_type)
        self.lr = tf.compat.v1.placeholder(self.tf_float_type)

        self.X = tf.compat.v1.placeholder(self.tf_float_type, shape=[self.n, self.d])
        W = tf.Variable(tf.zeros([self.d, self.d], self.tf_float_type))

        self.W_prime = self._preprocess_graph(W)
        self.mse_loss = self._get_mse_loss(self.X, self.W_prime)
        self.mle = self._get_mle_loss()

        self.h = tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d    # Acyclicity

        self.loss = self.mle \
                    + self.l1_lambda * tf.norm(self.W_prime, ord=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _preprocess_graph(self, W):
        # Mask the diagonal entries of graph
        return tf.linalg.set_diag(W, tf.zeros(W.shape[0], dtype=self.tf_float_type))

    def _get_mse_loss(self, X, W_prime):
        X_prime = tf.matmul(X, W_prime)
        return tf.square(tf.linalg.norm(X - X_prime))

    
    def _get_mle_loss(self):

        # Equal-scale version
        # it remains a challenge to address the non-equal noise variance case, see page 63 in https://www.ml.cmu.edu/research/phd-dissertation-pdfs/thesis-zheng-xun.pdf
        sigma = tf.math.sqrt(
                tf.math.reduce_sum(tf.square(self.X - self.X @ self.W_prime)) / (self.n * self.d)
                )

        # Standardize the data
        s = (self.X - self.X @ self.W_prime) / sigma

        if self.MLEScore == 'Sup-G':
            nm_term = 2/self.n * tf.reduce_sum(
                    tf.math.log(
                        tf.math.cosh(s)
                        )
                    )

        elif self.MLEScore == 'Sub-G':
            nm_term = -1 / self.n * tf.reduce_sum(
                tf.math.log(
                    tf.math.cosh(s)
                ) + tf.math.square(s) / 2
            )
        else:
            raise ValueError("Unknown Score.")

        return nm_term + self.d * tf.math.log(sigma) - tf.linalg.slogdet(tf.eye(self.d) - self.W_prime)[1]