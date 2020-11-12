# CodeTransformer

CodeTransformer is a code search engine that uses Deep Learning to answer natural language queries.

The Diploma Thesis in Greek can be accessed [here](https://ikee.lib.auth.gr/record/320405/files/Evangelos_Papathomas_8692_CODEtransformer.pdf).

## Preprocessing

To preprocess the dataset, run the following command.

```bash
python preprocess.py
```

## Training

To train the model, run the following command.

```bash
python train.py
```

## Evaluation

To evaluate the model, run the following command.

```bash
python evaluate.py
```

CodeTransformer was evaluated using the 99 evaluation queries suggested by [CodeSearchNet](https://github.com/github/CodeSearchNet) and the 40 most popular Java-related questions on Stack Overflow. The recommended code snippets for the CodeSearchNet evaluation queries were obtained through _evaluation.py_, while the recommended code snippets for the Stack Oveflow questions were obtained manually through CodeTransformers web application. The results for both datasets are available in the _results_ folder.

**Libraries and APIs listed in /app/requirements.txt required.**