import unittest

from tfidf import tfIDF


class TestTFIDFSearch(unittest.TestCase):
    def setUp(self):
        self.dataset = ['my name is tfidf',
                        'her name is feature extraction',
                        'his name is fancy']

    def test_search_returns_ranked_results(self):
        tfidf_object = tfIDF(self.dataset)
        results = tfidf_object.search('name feature extraction')
        idx = results[0][0]
        self.assertEqual(self.dataset[idx], 'her name is feature extraction')

    def test_search_readme_example(self):
        tfidf_object = tfIDF(['my name is tfidf', 'her name is feature extraction'])
        results = tfidf_object.search('tfidf')
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], 0.0)
        self.assertEqual(results[1][1], 0.0)

    def test_search_matches_whole_tokens_not_substrings(self):
        # 'am' is a substring of 'name' but a different token, so the
        # document holding 'am' must not score against the query 'name'
        tfidf_object = tfIDF(['am minute', 'name game'])
        results = tfidf_object.search('name')
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[1][1], 0.0)

    def test_search_is_case_insensitive(self):
        tfidf_object = tfIDF(['Herbal Formulation',
                              'MicroRNAs and cancer resistance'])
        results = tfidf_object.search('micrornas')
        self.assertEqual(results[0][0], 1)
        self.assertGreater(results[0][1], 0.0)

    def test_search_rejects_non_string_query(self):
        tfidf_object = tfIDF(self.dataset)
        with self.assertRaises(TypeError):
            tfidf_object.search(123)


class TestTFIDFValidation(unittest.TestCase):
    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            tfIDF([])

    def test_non_list_dataset_rejected(self):
        with self.assertRaises(TypeError):
            tfIDF('my name is tfidf')

    def test_non_string_document_rejected(self):
        with self.assertRaises(TypeError):
            tfIDF(['my name is tfidf', 42])

    def test_tuple_dataset_accepted(self):
        tfidf_object = tfIDF(('my name is tfidf', 'her name is fancy'))
        self.assertEqual(len(tfidf_object.getSCORE()), 2)


class TestTFIDFValues(unittest.TestCase):
    def setUp(self):
        # the two example documents used in TFIDF.ipynb
        self.tfidf_object = tfIDF([
            'Herbal Formulation Against Dental Caries Causing Microorganisms '
            'Using Extracts of Stevia Rebaudiana Leaves (A Natural Sweetner)',
            'MicroRNAs and cancer resistance: A new molecular plot'])

    def test_idf_values(self):
        self.assertAlmostEqual(self.tfidf_object.getIDF()['new'],
                               0.3010299956639812, places=6)

    def test_score_values(self):
        scores = self.tfidf_object.getSCORE()
        self.assertAlmostEqual(scores[0], 4.816479930623699, places=6)
        self.assertAlmostEqual(scores[1], 2.4082399653118496, places=6)

    def test_search_notebook_query(self):
        results = self.tfidf_object.search('molecular pathogenesis micrornas')
        self.assertEqual(results[0][0], 1)
        self.assertGreater(results[0][1], 0.0)
        self.assertEqual(results[1][1], 0.0)


class TestTFIDFEncapsulation(unittest.TestCase):
    # getters must hand out copies, not internal state that callers
    # could corrupt and skew every later search
    def setUp(self):
        self.tfidf_object = tfIDF(['my name is tfidf', 'her name is fancy'])

    def test_wordset_copy(self):
        self.tfidf_object.getWORDSET().add('tamper')
        self.assertNotIn('tamper', self.tfidf_object.getWORDSET())

    def test_worddict_copy(self):
        self.tfidf_object.getWORDDICT()[0]['name'] = 999
        self.assertNotEqual(self.tfidf_object.getWORDDICT()[0]['name'], 999)

    def test_idf_copy(self):
        self.tfidf_object.getIDF()['name'] = 999.0
        self.assertNotEqual(self.tfidf_object.getIDF()['name'], 999.0)

    def test_tfidf_copy(self):
        self.tfidf_object.getTFIDF()[0]['tamper'] = 999.0
        self.assertNotIn('tamper', self.tfidf_object.getTFIDF()[0])


if __name__ == '__main__':
    unittest.main()
