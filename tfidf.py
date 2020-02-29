from operator import itemgetter
class tfIDF():
    def __init__(self,doc):
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
        import math
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
        return list(tfidf_out)

    def getSCORE(self):
        score = []
        for u in range(0,len(self.tfidf)):
            score.append(0)
            for i in self.tfidf[u]:
                score[u] += self.tfidf[u][i]
        return score
    
    def getIDF(self):
        return self.idf

    def getWORDSET(self):
        return self.wordSet
    
    def getWORDDICT(self):
        return self.wordDict
    
    def search(self,query):
        query_tfidf = []

        num = 0
        tfidf_out = self.getTFIDF()
        for i in tfidf_out:
            query_tfidf.append(0.0)
            for u in i:
                if u in query:
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