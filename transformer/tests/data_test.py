
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tensorflow as tf
from data import NMTDataset

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, "test_data")

nmtdataset = NMTDataset(
    'en','vi', './'
)

max_length = 10
class NMTDatasetTest(tf.test.TestCase):
  def test_preprocess_sentence(self):
    self.assertEqual(
        nmtdataset.preprocess_sentence(
            'I love you', max_length
        ),
        '<start> i love you <end>'
    )



if __name__ == "__main__":
  tf.test.main()
