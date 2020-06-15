import torch

class Vocab():
    def __init__(self,embed,word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}
        assert len(self.word2id) == len(self.id2word) #nếu độ dài 2 cái dictionary này mà khác nhau thì báo lỗi
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN' #token thêm vào để đánh dấu bắt đầu và kết thúc câu
        self.UNK_TOKEN = 'UNK_TOKEN' #token lạ
    
    def __len__(self):
        return len(word2id)

    def i2w(self,idx):
        return self.id2word[idx]
    def w2i(self,w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX
    
    def make_features(self,batch,sent_trunc=50,doc_trunc=100,split_token='\n'): #trunc = truncate: cắt ngắn
        sents_list,targets,doc_lens = [],[],[]
        # trunc document
        for doc,label in zip(batch['doc'],batch['labels']): #batch là 1 dictionay với khóa là 'doc' tương tự 'labels'
            sents = doc.split(split_token) #cắt doc thành các sent
            labels = label.split(split_token) #cắt các nhãn
            labels = [int(l) for l in labels] #chuyển nhãn thành số label là 0 haowjc 1
            max_sent_num = min(doc_trunc,len(sents)) #số câu tối đa trong 1 batch document
            sents = sents[:max_sent_num] #cắt câu sao cho phù hợp với tham số truyền vào
            labels = labels[:max_sent_num] #tương tự cắt các nhãn của câu tương ứng
            sents_list += sents #thêm các câu vào danh sách sent --- phép cộng thêm phần tử vào list
            targets += labels #danh sách các nhãn của các câu (xác suất câu đấy được trích tóm tắt)
            doc_lens.append(len(sents)) # danh sách độ dài các câu trên 1 batch doc sample.
            
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = [] #danh sách các câu
        for sent in sents_list: #xét từng câu trong danh sách câu
            #sent có dạng là 'word1 word2 word3'
            #sents_list có dạng là ['word1 word2 word3','word1 word2 word3','word1 word2 word3']
            words = sent.split() #word = ['word1','word2'] do phương thức là split()
            if len(words) > sent_trunc: #nếu số từ nhiều hơn độ dài câu
                words = words[:sent_trunc] #thì cắt phần cuối từ đi
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            #max_sent_len tối đa là 50
            batch_sents.append(words) #gán cụm từ vào 1 list mối cụm từ được lưu như 1 list
        #max_len_sent là độ dài câu dài nhất.
        features = []
        for sent in batch_sents: #xét các cụm câu
            # mỗi 1 word có PAD_IDX
            # mỗi 1 từ sẽ được gắn thêm phần PADDING để đồng bộ độ dài là max_sent_len.
            # PADDing đặt là 1 ko phải là 0 tùy từng bài
            # phép cộng bên dưới là nối chuỗi vì kiểu dữ liệu là list
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature) #list của các list feature
            # features là 1 list có phần tử là các vị trí của từ (câu đang xét) trong tập từ điển - dạng list
        features = torch.LongTensor(features)   #chuyển list thành tensor kiểu long (matrix)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries'] #tập output là phần các câu trích xuất

        return features,targets,summaries,doc_lens
    #gán đặc trưng cho tập data của người dùng
    def make_predict_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='. '):
        sents_list, doc_lens = [],[]
        for doc in batch:
            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc,len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)

        return features, doc_lens