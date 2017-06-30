import tensorflow as tf
from tensorflow.python.platform import googletest
from tffm.fm_ops import fm_ops


class FmParserOpTest(tf.test.TestCase):
    EXAMPLES = [
        "1 1:1 2:2 3:3 4:4",
        "-1 5:1 6:1 7:1 ",
        "1.0 8:0.1 9:0.2 10:0.3"]
    VOCAB_SIZE = 10000
    TARGET_SIZES = [4, 3, 3]
    TARGET_LABELS = [1, -1, 1.0]
    TARGET_FEATURE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TARGET_FEATURE_VALS = [1, 2, 3, 4, 1, 1, 1, 0.1, 0.2, 0.3]

    def testNoHash(self):
        parser_op = fm_ops.fm_parser(
            tf.constant(
                self.EXAMPLES,
                tf.string),
            self.VOCAB_SIZE)
        with self.test_session() as sess:
            labels, sizes, feature_ids, feature_vals = sess.run(parser_op)
            self.assertAllClose(self.TARGET_LABELS, labels)
            self.assertAllEqual(self.TARGET_FEATURE_IDS, feature_ids)
            self.assertAllClose(self.TARGET_FEATURE_VALS, feature_vals)
            self.assertAllEqual(self.TARGET_SIZES, sizes)

    def testWithHash(self):
        parser_op = fm_ops.fm_parser(
            tf.constant(
                self.EXAMPLES,
                tf.string),
            self.VOCAB_SIZE,
            True)
        string_ids = [str(x) for x in self.TARGET_FEATURE_IDS]
        hashed_feature_ids = tf.string_to_hash_bucket(
            string_ids, self.VOCAB_SIZE)
        with self.test_session() as sess:
            hashed_ids = sess.run(hashed_feature_ids)
            labels, sizes, feature_ids, feature_vals = sess.run(parser_op)
            self.assertAllClose(labels, self.TARGET_LABELS)
            self.assertAllEqual(sizes, self.TARGET_SIZES)
            self.assertAllEqual(feature_ids, hashed_ids)
            self.assertAllClose(feature_vals, self.TARGET_FEATURE_VALS)

    def testError(self):
        with self.test_session() as sess:
            with self.assertRaisesRegexp(
                tf.errors.InvalidArgumentError,
                "Label could not be read in example: "
            ):
                sess.run(
                    fm_ops.fm_parser(
                        tf.constant(["one 1:1 2:2 3:3 4:4"], tf.string),
                        self.VOCAB_SIZE, False
                    )
                )

            with self.assertRaisesRegexp(
                tf.errors.InvalidArgumentError,
                "Invalid format in example: "
            ):
                sess.run(
                    fm_ops.fm_parser(
                        tf.constant(["1 one:1 2:2 3:3 4:4"], tf.string),
                        self.VOCAB_SIZE, False
                    )
                )

            with self.assertRaisesRegexp(
                tf.errors.InvalidArgumentError,
                "Invalid feature value. "
            ):
                sess.run(
                    fm_ops.fm_parser(
                        tf.constant(["1 1:one 2:2 3:3 4:4"], tf.string),
                        self.VOCAB_SIZE, False
                    )
                )


if __name__ == "__main__":
    googletest.main()
