from datautils import load_unsup_dataset, avg_collapse, get_context_idx, extract_context_pairs, custom_collate
from torch.utils.data import DataLoader
from Models import SkipGram


def test_unsup():
    '''
    Uses MOSI dataset to test if loading unsupervised dataset is okay to go
    '''
    text_field = 'CMU_MOSI_ModifiedTimestampedWords'
    visual_field = 'CMU_MOSI_OpenFace'
    acoustic_field = 'CMU_MOSI_openSMILE_IS09'

    dataset, w2i = load_unsup_dataset(text_field, visual_field, acoustic_field, collapse_function=avg_collapse)
    return dataset, w2i


def test_get_context_idx():
    idx = [3, 8, 10, 10]
    context_size = [5, 4, 5, 12]
    total_len = [6, 16, 20, 15]
    for config in zip(idx, context_size, total_len):
        print(get_context_idx(*config))
    return True


def test_context_pairs(dataset, context_size):
    return extract_context_pairs(dataset, context_size)


def test_collate(training_data):
    dataloader = DataLoader(training_data, batch_size=5)
    return dataloader


def test_model():
    model = SkipGram(30, 5, 25, 25)
    word2id = {'four': 4, 'three': 3, 'two': 2, 'one': 1, 'zero': 0}
    id2word = {v: k for k, v in word2id.items()}
    model.save_embedding(id2word, 'resources/test.txt')


if __name__ == "__main__":
    dataset, w2i = test_unsup()
    test_get_context_idx()
    pairs = test_context_pairs(dataset, 2)
    dataloader = test_collate(pairs)
    test_model()

