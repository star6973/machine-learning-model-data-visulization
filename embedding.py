from scipy.sparse import csr_matrix
import time
from pickle import dump, load

def pre_train_embedding(embed_type, pre_embed_model, train, test):
    # x열은 토크나이징 된 단어들 목록
    # y열은 타겟 라벨

    # counter vectorize, tf-idf용 corpus // 빈칸으로 띄어쓰기
    train_corpus1 = []
    # word2vec, doc2vec용 corpus # ,로 띄어쓰기와 리스트 형식
    train_corpus2 = []

    for words in train['x']:
        sentence1 = ""
        sentence2 = []

        for word in words.split(","):
            sentence1 += word + " "
            sentence2.append(word)

        sentence1 = sentence1[:len(sentence1) - 1]

        train_corpus1.append(sentence1)
        train_corpus2.append(sentence2)

    # counter vectorize, tf-idf용 corpus // 빈칸으로 띄어쓰기
    test_corpus1 = []
    # word2vec, doc2vec용 corpus # ,로 띄어쓰기와 리스트 형식
    test_corpus2 = []

    for words in test['x']:
        sentence1 = ""
        sentence2 = []

        for word in words.split(","):
            sentence1 += word + " "
            sentence2.append(word)

        sentence1 = sentence1[:len(sentence1) - 1]

        test_corpus1.append(sentence1)
        test_corpus2.append(sentence2)

    if embed_type == "CounterVector":

        start = time.time()

        count_vectorizer = load(
            open("/home/ubuntu/project2/embedding_model/" + pre_embed_model, "rb"))

        count_train_vectors = count_vectorizer.transform(train_corpus1)
        count_test_vectors = count_vectorizer.transform(test_corpus1)

        sparse_count_train_x = csr_matrix(count_train_vectors)
        sparse_count_test_x = csr_matrix(count_test_vectors)

        end = time.time()

        print('pre-train CounterVectorizer embedding time: {}'.format(end - start))

        return sparse_count_train_x, sparse_count_test_x, train['y'].values, test['y'].values

    elif embed_type == "TF-IDF":

        start = time.time()

        tfidf_vectorizer = load(
            open("/home/ubuntu/project2/embedding_model/" + pre_embed_model, "rb"))

        tf_train_vectors = tfidf_vectorizer.transform(train_corpus1)
        tf_test_vectors = tfidf_vectorizer.transform(test_corpus1)

        sparse_tf_train_x = csr_matrix(tf_train_vectors)
        sparse_tf_test_x = csr_matrix(tf_test_vectors)

        end = time.time()

        print('pre-train TfidfVectorizer embedding time: '.format(end - start))

        return sparse_tf_train_x, sparse_tf_test_x, train['y'].values, test['y'].values

    elif embed_type == "Doc2Vec":

        start = time.time()

        from collections import namedtuple
        TaggedDocument = namedtuple('TaggedDocument', 'words tags')

        doc2vec_train_tag = [TaggedDocument(doc, tag) for doc, tag in zip(train_corpus2, train['y'].values)]
        doc2vec_test_tag = [TaggedDocument(doc, tag) for doc, tag in zip(test_corpus2, test['y'].values)]

        from gensim.models.doc2vec import Doc2Vec

        doc_vectorizer = Doc2Vec.load(
            '/home/ubuntu/project2/embedding_model/' + pre_embed_model)

        for epoch in range(10):
            doc_vectorizer.train(doc2vec_train_tag, total_examples=doc_vectorizer.corpus_count, epochs=10)
            doc_vectorizer.alpha -= 0.002  # decrease the learning rate
            doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay

        doc_train_vectors = [doc_vectorizer.infer_vector(doc.words) for doc in doc2vec_train_tag]
        doc_train_tags = [doc.tags for doc in doc2vec_train_tag]

        doc_test_vectors = [doc_vectorizer.infer_vector(doc.words) for doc in doc2vec_test_tag]
        doc_test_tags = [doc.tags for doc in doc2vec_test_tag]

        import numpy as np

        doc_train_vectors_np = np.array(doc_train_vectors)
        doc_train_tags_np = np.array(doc_train_tags)

        doc_test_vectors_np = np.array(doc_test_vectors)
        doc_test_tags_np = np.array(doc_test_tags)

        sparse_doc_train_x = csr_matrix(doc_train_vectors_np)
        sparse_doc_test_x = csr_matrix(doc_test_vectors_np)

        end = time.time()

        print('pre-train Doc2Vec embedding time: {}'.format(end - start))

        return sparse_doc_train_x, sparse_doc_test_x, doc_train_tags_np, doc_test_tags_np

    elif embed_type == "user_defined_embedding":
        pass


def embedding(filename, embed_type, train, test, embed_params):
    # x열은 토크나이징 된 단어들 목록
    # y열은 타겟 라벨

    # counter vectorize, tf-idf용 corpus // 빈칸으로 띄어쓰기
    train_corpus1 = []
    # word2vec, doc2vec용 corpus # ,로 띄어쓰기와 리스트 형식
    train_corpus2 = []

    for words in train['x']:
        sentence1 = ""
        sentence2 = []

        for word in words.split(","):
            sentence1 += word + " "
            sentence2.append(word)

        sentence1 = sentence1[:len(sentence1) - 1]

        train_corpus1.append(sentence1)
        train_corpus2.append(sentence2)

    # counter vectorize, tf-idf용 corpus // 빈칸으로 띄어쓰기
    test_corpus1 = []
    # word2vec, doc2vec용 corpus # ,로 띄어쓰기와 리스트 형식
    test_corpus2 = []

    for words in test['x']:
        sentence1 = ""
        sentence2 = []

        for word in words.split(","):
            sentence1 += word + " "
            sentence2.append(word)

        sentence1 = sentence1[:len(sentence1) - 1]

        test_corpus1.append(sentence1)
        test_corpus2.append(sentence2)

    if embed_type == "CounterVector":

        start = time.time()

        from sklearn.feature_extraction.text import CountVectorizer

        count_vectorizer = CountVectorizer(min_df=embed_params['min_df'],
                                           max_df=embed_params['max_df'],
                                           max_features=embed_params['max_features'])

        count_train_vectors = count_vectorizer.fit_transform(train_corpus1)
        count_test_vectors = count_vectorizer.transform(test_corpus1)

        dump(count_vectorizer,
             open("/home/ubuntu/project2/embedding_model/" + filename + "_counterVec.pkl", "wb"))

        sparse_count_train_x = csr_matrix(count_train_vectors)
        sparse_count_test_x = csr_matrix(count_test_vectors)

        end = time.time()

        print('CounterVectorizer embedding time: {}'.format(end - start))

        return sparse_count_train_x, sparse_count_test_x, train['y'].values, test['y'].values

    elif embed_type == "TF-IDF":

        start = time.time()

        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(min_df=embed_params['min_df'],
                                           max_df=embed_params['max_df'],
                                           max_features=embed_params['max_features'])

        tf_train_vectors = tfidf_vectorizer.fit_transform(train_corpus1)
        tf_test_vectors = tfidf_vectorizer.transform(test_corpus1)

        dump(tfidf_vectorizer,
             open("/home/ubuntu/project2/embedding_model/" + filename + "_tfidf.pkl", "wb"))

        sparse_tf_train_x = csr_matrix(tf_train_vectors)
        sparse_tf_test_x = csr_matrix(tf_test_vectors)

        end = time.time()

        print('TfidfVectorizer embedding time: '.format(end - start))

        return sparse_tf_train_x, sparse_tf_test_x, train['y'].values, test['y'].values

    elif embed_type == "Doc2Vec":

        start = time.time()

        from collections import namedtuple
        TaggedDocument = namedtuple('TaggedDocument', 'words tags')

        doc2vec_train_tag = [TaggedDocument(doc, tag) for doc, tag in zip(train_corpus2, train['y'].values)]
        doc2vec_test_tag = [TaggedDocument(doc, tag) for doc, tag in zip(test_corpus2, test['y'].values)]

        from gensim.models.doc2vec import Doc2Vec

        doc_vectorizer = Doc2Vec(documents=doc2vec_train_tag,
                                 dm=embed_params['dm'],
                                 alpha=embed_params['alpha'],
                                 vector_size=embed_params['vector_size'],
                                 window=embed_params['window'],
                                 negative=embed_params['negative'],
                                 epochs=embed_params['epochs'])

        doc_train_vectors = [doc_vectorizer.infer_vector(doc.words) for doc in doc2vec_train_tag]
        doc_train_tags = [doc.tags for doc in doc2vec_train_tag]

        doc_test_vectors = [doc_vectorizer.infer_vector(doc.words) for doc in doc2vec_test_tag]
        doc_test_tags = [doc.tags for doc in doc2vec_test_tag]

        doc_vectorizer.save(
            '/home/ubuntu/project2/embedding_model/' + filename + '_doc2vec.pkl')

        import numpy as np

        doc_train_vectors_np = np.array(doc_train_vectors)
        doc_train_tags_np = np.array(doc_train_tags)

        doc_test_vectors_np = np.array(doc_test_vectors)
        doc_test_tags_np = np.array(doc_test_tags)

        sparse_doc_train_x = csr_matrix(doc_train_vectors_np)
        sparse_doc_test_x = csr_matrix(doc_test_vectors_np)

        end = time.time()

        print('Doc2Vec embedding time: {}'.format(end - start))

        return sparse_doc_train_x, sparse_doc_test_x, doc_train_tags_np, doc_test_tags_np

    elif embed_type == "user_defined_embedding":
        pass