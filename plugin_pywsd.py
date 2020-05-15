#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepnlpf.core.boost import Boost
from deepnlpf.core.iplugin import IPlugin
from deepnlpf.core.output_format import OutputFormat

from pywsd import disambiguate

class Plugin(IPlugin):

    def __init__(self, id_pool, lang, document, pipeline):
        self._id_pool = id_pool
        self._document = document
        self._pipeline = pipeline

    def run(self):
        option = {}

        if "wsd" in self._pipeline:
            option['wsd'] = Boost().multithreading(self.wsd, self._document['sentences'])

        return option['wsd']

    def wrapper(self):
        pass

    def wsd(self, sentence):
        result = disambiguate(sentence)

        list_item = []

        for item in result:
            item = {
                "word": item[0],
                "synset": str(item[1]).replace("Synset('", "").replace("')", "")
            }

            list_item.append(item)

        return list_item

    def disambiguate_max(self, sentence):
        from pywsd.similarity import max_similarity as maxsim
        return disambiguate(sentence, algorithm=maxsim, similarity_option='wup', keepLemmas=True)

    def disambiguate_lesk(self, sentence, ambiguous, pos):
        """
            @param sentence : I went to the bank to deposit my money

            @param ambiguous : bank

            @param pos : n

            @return : Synset('depository_financial_institution.n.01')
        """
        from pywsd.lesk import simple_lesk
        return simple_lesk(sentence, ambiguous, pos)

    def definition(self, answer):
        """
            @param answer : Synset('depository_financial_institution.n.01')

            @return : a financial institution that accepts deposits and channels the money into lending activities
        """
        return answer.definition()

    def out_format(self, annotation):
        pass
