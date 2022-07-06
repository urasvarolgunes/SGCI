from nltk.corpus import wordnet as wn

dic = {"google": "GoogleNews-vectors-negative300.txt",
        "glove": "glove.840B.300d.txt",
        "fast": "wiki.en.vec"}

def read_embedding(name):
    with open("../raw_embedding/"+dic[name], 'r') as f:
        preembed = f.read().split('\n')
        if len(preembed[-1]) < 2:
            del preembed[-1]
    preembed = [e.split(' ', 1) for e in preembed]
    words = [e[0] for e in preembed]
    embed = [e[1] for e in preembed]
    return dict(zip(words,embed))


def build_data(names):
    embed = [read_embedding(n) for n in names]
    words = set([w for w in wn.words()])
    word_sets = [set(e.keys()) for e in embed]
    word_sets.append(words)
    words = set.intersection(*word_sets)
    with open("word_list.txt", 'w') as f:
        f.write('\n'.join(words))
        print("Word list saved.")
    embed = [[e[w] for w in words] for e in embed]
    for i in range(len(names)):
        with open(names[i]+".txt", 'w') as f:
            f.write('\n'.join(embed[i]))
            print(names[i]+" saved.")


if __name__ == "__main__":
    build_data(list(dic.keys()))
