import math
from operator import itemgetter

class tfIDF():
    def __init__(self,doc):
        if not isinstance(doc, (list, tuple)):
            raise TypeError('doc must be a list of strings')
        if len(doc) == 0:
            raise ValueError('doc must contain at least one document')
        for d in doc:
            if not isinstance(d, str):
                raise TypeError('every document must be a string')
        self.doc = [x.split() for x in doc]
        self.wordSet = self.getWordset(self.doc)
        self.wordDict = self.getWorddict(self.doc)
        self.idf = self.computeIDF(self.wordDict)
        self.tfidf = []

    def getWordset(self,doc):
        wordSet = []
        for x in doc:
            wordSet = wordSet + x
        return set(wordSet)

    def getWorddict(self,doc):
        wordDict = []
        for i in range(0,len(doc)):
            wordDict.append(dict.fromkeys(self.wordSet, 0))
        for i in range(0,len(wordDict)):
            for word in doc[i]:
                wordDict[i][word]+=1
        return wordDict

    def computeIDF(self,docList):
        idfDict = {}
        N = len(docList)

        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for doc in docList:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1

        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))

        return idfDict    
    
    def computeTFIDF(self, i, idfs):
        tfidf = {}
        for word in self.doc[i]:
            tfidf[word] = self.wordDict[i][word]*idfs[word]
        return tfidf
    
    def getTFIDF(self):
        idfs = self.idf
        tfidf_out = []
        for x in range(0,len(self.doc)):
            tfidf_out.append(self.computeTFIDF(x, idfs))
        self.tfidf = tfidf_out
        return [dict(scores) for scores in tfidf_out]

    def getSCORE(self):
        # make sure scores exist even if getTFIDF was never called
        self.getTFIDF()
        score = []
        for u in range(0,len(self.tfidf)):
            score.append(0)
            for i in self.tfidf[u]:
                score[u] += self.tfidf[u][i]
        return score

    def getIDF(self):
        return dict(self.idf)

    def getWORDSET(self):
        return set(self.wordSet)

    def getWORDDICT(self):
        return [dict(counts) for counts in self.wordDict]

    def search(self,query):
        if not isinstance(query, str):
            raise TypeError('query must be a string')
        # match on whole tokens, case-insensitively; a plain substring check
        # would let partial tokens (e.g. 'am' inside 'name') score documents
        query_tokens = set(query.lower().split())

        query_tfidf = []

        num = 0
        tfidf_out = self.getTFIDF()
        for i in tfidf_out:
            query_tfidf.append(0.0)
            for u in i:
                if u.lower() in query_tokens:
                    query_tfidf[num] += i[u]
            num = num + 1
        
        hasil = [a*b for a,b in zip(query_tfidf,self.getSCORE())]
        for i in range(0,len(hasil)):
            hasil[i] = [i, hasil[i]]
        
        hasil_akhir = reversed(sorted(hasil, key=itemgetter(1)))
        hasil_akhir = [x for x in hasil_akhir]
        
        return hasil_akhir

# dataset = ["Herbal Formulation Against Dental Caries Causing Microorganisms Using Extracts of Stevia Rebaudiana Leaves (A Natural Sweetner)","MicroRNAs and cancer resistance: A new molecular plot"] 
# tfidf = tfIDF(dataset)
# print(tfidf.search('molecular pathogenesis micrornas'))