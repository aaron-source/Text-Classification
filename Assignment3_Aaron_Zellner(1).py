{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Assignment3_Aaron_Zellner(1).ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lzo8YMyn04bR",
        "colab_type": "text"
      },
      "source": [
        "## Part 1\n",
        "\n",
        "The goal of this part is to explore some of the main [scikit-learn](http://scikit-learn.org/stable/index.html) tools on a single practical task: analysing a collection of text documents (newsgroups posts) on twenty different topics.\n",
        "\n",
        "In this section we will see how to:\n",
        "\n",
        "1. load the file contents and the categories\n",
        "2. extract feature vectors suitable for machine learning\n",
        "3. train a model to perform text classification\n",
        "4. evaluate the performance of the trained model\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xWLXr41H1kFz",
        "colab_type": "text"
      },
      "source": [
        "### Loading the 20 newsgroups dataset\n",
        "The 20 newsgroups dataset comprises around 20,000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). \n",
        "\n",
        "In the following we will use the built-in dataset loader for 20 newsgroups from scikit-learn.\n",
        "\n",
        "The **sklearn.datasets.fetch_20newsgroups** function is a data fetching / caching functions that downloads the data archive from the original 20 newsgroups website, extracts the archive contents in a local folder and calls the **sklearn.datasets.load_files** on either the training or testing set folder, or both of them. Here, we are loading only 4 categories."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cUpQ5bYq2GMK",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        },
        "outputId": "45b1d3c6-8e83-4403-8701-c30f06988c22"
      },
      "source": [
        "cats = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'] #select only 4 categories for fast running times\n",
        "\n",
        "from sklearn.datasets import fetch_20newsgroups\n",
        "newsgroups_train = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading 20news dataset. This may take a few minutes.\n",
            "Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "byNIxaqL2LyP",
        "colab_type": "text"
      },
      "source": [
        "We can now list the 4 categories as follows:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LZORgv8ozwtR",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "c3272e6d-745b-438a-bed6-c59e94a764c0"
      },
      "source": [
        "\n",
        "print(newsgroups_train.target_names)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "['alt.atheism', 'comp.graphics', 'sci.space', 'talk.religion.misc']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "d-q-D5aP6JuR",
        "colab_type": "text"
      },
      "source": [
        "The real data lies in the **filenames** and **target** attributes. The target attribute is the integer index of the category:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b4svhJW-6YEp",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "85e0a6fb-ae88-4ec3-bd2d-d7bba3fb09db"
      },
      "source": [
        "print(newsgroups_train.filenames[0]) #print the name of the first file\n",
        "print(newsgroups_train.target[0]) #print the category of the first example\n",
        "print(newsgroups_train.data[0]) #print the text of the first example"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/root/scikit_learn_data/20news_home/20news-bydate-train/comp.graphics/38816\n",
            "1\n",
            "Hi,\n",
            "\n",
            "I've noticed that if you only save a model (with all your mapping planes\n",
            "positioned carefully) to a .3DS file that when you reload it after restarting\n",
            "3DS, they are given a default position and orientation.  But if you save\n",
            "to a .PRJ file their positions/orientation are preserved.  Does anyone\n",
            "know why this information is not stored in the .3DS file?  Nothing is\n",
            "explicitly said in the manual about saving texture rules in the .PRJ file. \n",
            "I'd like to be able to read the texture rule information, does anyone have \n",
            "the format for the .PRJ file?\n",
            "\n",
            "Is the .CEL file format available from somewhere?\n",
            "\n",
            "Rych\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UUE3z92Z73xt",
        "colab_type": "text"
      },
      "source": [
        "### Converting text to vectors\n",
        "In order to feed machine learing models with the text data, one first need to turn the text into vectors of numerical values suitable for statistical analysis. This can be achieved with the utilities of the **sklearn.feature_extraction.text** as demonstrated in the following example.\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1Eb7r9SO8Y54",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "ce6976e1-e5f7-4fa8-a3a4-a01664333c09"
      },
      "source": [
        "from sklearn.feature_extraction.text import CountVectorizer #tokenizer\n",
        "vectorizer = CountVectorizer(stop_words='english') #remove english stop words\n",
        "vectors = vectorizer.fit_transform(newsgroups_train.data)\n",
        "print (vectors.shape) #print the size"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(2034, 26576)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hKjFkDVW9R6N",
        "colab_type": "text"
      },
      "source": [
        "### Training a machine learining model and evaluate its performance\n",
        " Let’s use a multinomial Naive Bayes classifier as discussed in the class."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2VLmrBPW9e4-",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "2617d132-9838-4766-e921-27f092e622e8"
      },
      "source": [
        "from sklearn.naive_bayes import MultinomialNB\n",
        "\n",
        "clf = MultinomialNB()\n",
        "clf.fit(vectors, newsgroups_train.target)\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IQXgKAlh96Ng",
        "colab_type": "text"
      },
      "source": [
        "Then let's print the F1 score on the test data."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qv_ljFnG9tVD",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 347
        },
        "outputId": "5f89e1b2-4e81-4f75-a837-ddd41e820518"
      },
      "source": [
        "from sklearn import metrics\n",
        "\n",
        "newsgroups_test = fetch_20newsgroups(subset='test', categories=cats, remove=('headers', 'footers', 'quotes'))  #test data\n",
        "vectors_test = vectorizer.transform(newsgroups_test.data)  #generate vectors from test data (using the same vectorizer)\n",
        "pred = clf.predict(vectors_test) #predict categories for the test data using the above trained classifier\n",
        "\n",
        "print(\"macro F1:\",metrics.f1_score(newsgroups_test.target, pred, average='macro'))\n",
        "print(\"micro F1:\",metrics.f1_score(newsgroups_test.target, pred, average='micro'))\n",
        "print(\"\\n\",metrics.classification_report(newsgroups_test.target, pred, target_names=newsgroups_test.target_names))\n",
        "cm = metrics.confusion_matrix(newsgroups_test.target, pred)\n",
        "print(\"Confusion Matrix:\\n\",cm)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "macro F1: 0.7564987834068887\n",
            "micro F1: 0.7804878048780488\n",
            "\n",
            "                     precision    recall  f1-score   support\n",
            "\n",
            "       alt.atheism       0.65      0.71      0.68       319\n",
            "     comp.graphics       0.91      0.90      0.91       389\n",
            "         sci.space       0.82      0.87      0.85       394\n",
            "talk.religion.misc       0.66      0.53      0.59       251\n",
            "\n",
            "          accuracy                           0.78      1353\n",
            "         macro avg       0.76      0.76      0.76      1353\n",
            "      weighted avg       0.78      0.78      0.78      1353\n",
            "\n",
            "Confusion Matrix:\n",
            " [[226   7  28  58]\n",
            " [ 11 352  23   3]\n",
            " [ 24  19 344   7]\n",
            " [ 87   8  22 134]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "We7-mPb_CSDq",
        "colab_type": "text"
      },
      "source": [
        "Let’s take a look at what the most informative features are:"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QlWPqCedCbh3",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 87
        },
        "outputId": "77b96bee-e8df-468a-84b0-70251dba45e2"
      },
      "source": [
        "import numpy as np\n",
        "def show_top10(classifier, vectorizer, categories):\n",
        "  feature_names = np.asarray(vectorizer.get_feature_names())\n",
        "  for i, category in enumerate(categories):\n",
        "    top10 = np.argsort(classifier.coef_[i])[-10:]\n",
        "    print(\"%s: %s\" % (category, \" \".join(feature_names[top10])))\n",
        "\n",
        "show_top10(clf, vectorizer, newsgroups_train.target_names)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "alt.atheism: like believe say atheism does just think don people god\n",
            "comp.graphics: software images files data use file jpeg edu graphics image\n",
            "sci.space: just shuttle time orbit data like earth launch nasa space\n",
            "talk.religion.misc: know say christian think just bible don jesus people god\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SXgUMEq_uDpi",
        "colab_type": "text"
      },
      "source": [
        "Instead of train-test setup, you can also perform cross-validation (CV). Following code shows CV results using the train set (although you could do this with the complete dataset)."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PPKe9JDSmCCT",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "e1e355a8-8652-4426-98b4-96a88b3ea265"
      },
      "source": [
        "from sklearn.model_selection  import cross_val_score\n",
        "cv_scores = cross_val_score(clf, vectors, newsgroups_train.target , cv=10, scoring=\"f1_macro\" )\n",
        "print(\"Avg. macro F1:\", np.mean(cv_scores))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Avg. macro F1: 0.8128198197572788\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DVCZERLPdyYy",
        "colab_type": "text"
      },
      "source": [
        "## Part 2\n",
        "\n",
        "Ths goal of this part is to write your own code to train a model to classify the given test dataset using part 1 as inspiriation.\n",
        "\n",
        "First, upload the given dataset (\"diseases-train.csv\") using the following cell. It contains 900 scientific artciles (identified by their PubMed IDs) and their labels. This is a multi-class problem.\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_ru8k_nK05xu",
        "colab_type": "code",
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCkgewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwogICAgICBwZXJjZW50LnRleHRDb250ZW50ID0KICAgICAgICAgIGAke01hdGgucm91bmQoKHBvc2l0aW9uIC8gZmlsZURhdGEuYnl0ZUxlbmd0aCkgKiAxMDApfSUgZG9uZWA7CiAgICB9CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 72
        },
        "outputId": "e4dc67bd-5cd3-46f1-fd90-193aae5351e1"
      },
      "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-bff9ae10-dfb2-46a9-84b2-a4e8f1edfe04\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-bff9ae10-dfb2-46a9-84b2-a4e8f1edfe04\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving diseases-train.csv to diseases-train (1).csv\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "B5jGiZKlW9Xa",
        "colab_type": "text"
      },
      "source": [
        "Then load the CSV file using pandas as below. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "80IHoQzj42MC",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "df_train = pd.read_csv(\"diseases-train.csv\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9wijrc25Y_tt",
        "colab_type": "text"
      },
      "source": [
        "Then you can iterate over the lines as follows. Each line has the format: pmid, category."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cyk6XDuCZITO",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "6e0c15e0-4d88-4a3d-8e39-14b57ecb1697"
      },
      "source": [
        "for index, row in df_train.iterrows():    \n",
        "    pmid = row[0]\n",
        "    print(pmid)\n",
        "    break"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "30141778\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "90r-99vPZTLh",
        "colab_type": "text"
      },
      "source": [
        "Then you can get the other information (i.e. title, abstract etc) associated with each of these articles using the [biopython](https://biopython.org/) library. First insatll the library as below."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5EV8JFxDgSoP",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 139
        },
        "outputId": "97a04a40-4fac-4b00-eef8-3b15c6dc55eb"
      },
      "source": [
        "!pip install biopython"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting biopython\n",
            "\u001b[?25l  Downloading https://files.pythonhosted.org/packages/76/02/8b606c4aa92ff61b5eda71d23b499ab1de57d5e818be33f77b01a6f435a8/biopython-1.78-cp36-cp36m-manylinux1_x86_64.whl (2.3MB)\n",
            "\u001b[K     |████████████████████████████████| 2.3MB 4.1MB/s \n",
            "\u001b[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from biopython) (1.18.5)\n",
            "Installing collected packages: biopython\n",
            "Successfully installed biopython-1.78\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3Udn-Lu9alvC",
        "colab_type": "text"
      },
      "source": [
        "You can fetch the information for an article with the **eftech** function as below. Find more information [here](http://biopython.org/DIST/docs/tutorial/Tutorial.html#sec:efetch). \n",
        "\n",
        "(Note: You can search for pubmed articles using keywords using [esearch](http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc123). This may be useful for your projects.)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uR1ekZbhat9r",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "fb761ecd-be45-4aca-a6fe-0b6d443ba888"
      },
      "source": [
        "from Bio import Entrez\n",
        "Entrez.email = \"A.N.Other@example.com\"  # Always tell NCBI who you are\n",
        "handle = Entrez.efetch(db=\"pubmed\", id=pmid, rettype=\"medline\", retmode=\"text\")\n",
        "print(handle.read())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "PMID- 30141778\n",
            "OWN - NLM\n",
            "STAT- MEDLINE\n",
            "DCOM- 20180904\n",
            "LR  - 20180904\n",
            "IS  - 1524-4725 (Electronic)\n",
            "IS  - 1076-0512 (Linking)\n",
            "VI  - 44\n",
            "IP  - 9\n",
            "DP  - 2018 Sep\n",
            "TI  - Inside \"Outside\" Job: Unexpected Geometric Skin Ulcerations Overlying Orthopedic \n",
            "      Hardware After Multimodal Laser Scar Revision.\n",
            "PG  - 1231-1233\n",
            "LID - 10.1097/DSS.0000000000001433 [doi]\n",
            "FAU - Borok, Jenna\n",
            "AU  - Borok J\n",
            "AD  - Division of Pediatric and Adolescent Dermatology, Rady Children's Hospital, San\n",
            "      Diego, California Department of Dermatology, University of California, Irvine\n",
            "      Irvine, California Department of Medicine, University of California, San Diego,\n",
            "      California DermOne, LLC, West Conshohocken, Pennsylvania.\n",
            "FAU - Ferris, Katherine\n",
            "AU  - Ferris K\n",
            "FAU - Vaux, Keith\n",
            "AU  - Vaux K\n",
            "FAU - Krakowski, Andrew C\n",
            "AU  - Krakowski AC\n",
            "LA  - eng\n",
            "PT  - Case Reports\n",
            "PT  - Journal Article\n",
            "PL  - United States\n",
            "TA  - Dermatol Surg\n",
            "JT  - Dermatologic surgery : official publication for American Society for Dermatologic\n",
            "      Surgery [et al.]\n",
            "JID - 9504371\n",
            "SB  - IM\n",
            "MH  - Cicatrix/*etiology/pathology/therapy\n",
            "MH  - Female\n",
            "MH  - Fibula/injuries\n",
            "MH  - Fracture Fixation, Internal/*adverse effects/*instrumentation\n",
            "MH  - Fractures, Bone/surgery\n",
            "MH  - Humans\n",
            "MH  - Internal Fixators/*adverse effects\n",
            "MH  - Laser Therapy/*adverse effects/methods\n",
            "MH  - Reoperation\n",
            "MH  - Skin Ulcer/*etiology/pathology/therapy\n",
            "EDAT- 2018/08/25 06:00\n",
            "MHDA- 2018/09/05 06:00\n",
            "CRDT- 2018/08/25 06:00\n",
            "PHST- 2018/08/25 06:00 [entrez]\n",
            "PHST- 2018/08/25 06:00 [pubmed]\n",
            "PHST- 2018/09/05 06:00 [medline]\n",
            "AID - 10.1097/DSS.0000000000001433 [doi]\n",
            "AID - 00042728-201809000-00013 [pii]\n",
            "PST - ppublish\n",
            "SO  - Dermatol Surg. 2018 Sep;44(9):1231-1233. doi: 10.1097/DSS.0000000000001433.\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0GMwj-otSIpT",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "dc660909-eab6-4759-b23f-1f164d0601e3"
      },
      "source": [
        "!pip install Biopython"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: Biopython in /usr/local/lib/python3.6/dist-packages (1.78)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from Biopython) (1.18.5)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lbIHHG8GSMrT",
        "colab_type": "code",
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCkgewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwogICAgICBwZXJjZW50LnRleHRDb250ZW50ID0KICAgICAgICAgIGAke01hdGgucm91bmQoKHBvc2l0aW9uIC8gZmlsZURhdGEuYnl0ZUxlbmd0aCkgKiAxMDApfSUgZG9uZWA7CiAgICB9CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "d0123abb-959c-4d1b-95e5-4f63742ae4ed"
      },
      "source": [
        "import nltk\n",
        "nltk.download('all')\n",
        "from google.colab import files\n",
        "files.upload()"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading collection 'all'\n",
            "[nltk_data]    | \n",
            "[nltk_data]    | Downloading package abc to /root/nltk_data...\n",
            "[nltk_data]    |   Package abc is already up-to-date!\n",
            "[nltk_data]    | Downloading package alpino to /root/nltk_data...\n",
            "[nltk_data]    |   Package alpino is already up-to-date!\n",
            "[nltk_data]    | Downloading package biocreative_ppi to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package biocreative_ppi is already up-to-date!\n",
            "[nltk_data]    | Downloading package brown to /root/nltk_data...\n",
            "[nltk_data]    |   Package brown is already up-to-date!\n",
            "[nltk_data]    | Downloading package brown_tei to /root/nltk_data...\n",
            "[nltk_data]    |   Package brown_tei is already up-to-date!\n",
            "[nltk_data]    | Downloading package cess_cat to /root/nltk_data...\n",
            "[nltk_data]    |   Package cess_cat is already up-to-date!\n",
            "[nltk_data]    | Downloading package cess_esp to /root/nltk_data...\n",
            "[nltk_data]    |   Package cess_esp is already up-to-date!\n",
            "[nltk_data]    | Downloading package chat80 to /root/nltk_data...\n",
            "[nltk_data]    |   Package chat80 is already up-to-date!\n",
            "[nltk_data]    | Downloading package city_database to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package city_database is already up-to-date!\n",
            "[nltk_data]    | Downloading package cmudict to /root/nltk_data...\n",
            "[nltk_data]    |   Package cmudict is already up-to-date!\n",
            "[nltk_data]    | Downloading package comparative_sentences to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package comparative_sentences is already up-to-\n",
            "[nltk_data]    |       date!\n",
            "[nltk_data]    | Downloading package comtrans to /root/nltk_data...\n",
            "[nltk_data]    |   Package comtrans is already up-to-date!\n",
            "[nltk_data]    | Downloading package conll2000 to /root/nltk_data...\n",
            "[nltk_data]    |   Package conll2000 is already up-to-date!\n",
            "[nltk_data]    | Downloading package conll2002 to /root/nltk_data...\n",
            "[nltk_data]    |   Package conll2002 is already up-to-date!\n",
            "[nltk_data]    | Downloading package conll2007 to /root/nltk_data...\n",
            "[nltk_data]    |   Package conll2007 is already up-to-date!\n",
            "[nltk_data]    | Downloading package crubadan to /root/nltk_data...\n",
            "[nltk_data]    |   Package crubadan is already up-to-date!\n",
            "[nltk_data]    | Downloading package dependency_treebank to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package dependency_treebank is already up-to-date!\n",
            "[nltk_data]    | Downloading package dolch to /root/nltk_data...\n",
            "[nltk_data]    |   Package dolch is already up-to-date!\n",
            "[nltk_data]    | Downloading package europarl_raw to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package europarl_raw is already up-to-date!\n",
            "[nltk_data]    | Downloading package floresta to /root/nltk_data...\n",
            "[nltk_data]    |   Package floresta is already up-to-date!\n",
            "[nltk_data]    | Downloading package framenet_v15 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package framenet_v15 is already up-to-date!\n",
            "[nltk_data]    | Downloading package framenet_v17 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package framenet_v17 is already up-to-date!\n",
            "[nltk_data]    | Downloading package gazetteers to /root/nltk_data...\n",
            "[nltk_data]    |   Package gazetteers is already up-to-date!\n",
            "[nltk_data]    | Downloading package genesis to /root/nltk_data...\n",
            "[nltk_data]    |   Package genesis is already up-to-date!\n",
            "[nltk_data]    | Downloading package gutenberg to /root/nltk_data...\n",
            "[nltk_data]    |   Package gutenberg is already up-to-date!\n",
            "[nltk_data]    | Downloading package ieer to /root/nltk_data...\n",
            "[nltk_data]    |   Package ieer is already up-to-date!\n",
            "[nltk_data]    | Downloading package inaugural to /root/nltk_data...\n",
            "[nltk_data]    |   Package inaugural is already up-to-date!\n",
            "[nltk_data]    | Downloading package indian to /root/nltk_data...\n",
            "[nltk_data]    |   Package indian is already up-to-date!\n",
            "[nltk_data]    | Downloading package jeita to /root/nltk_data...\n",
            "[nltk_data]    |   Package jeita is already up-to-date!\n",
            "[nltk_data]    | Downloading package kimmo to /root/nltk_data...\n",
            "[nltk_data]    |   Package kimmo is already up-to-date!\n",
            "[nltk_data]    | Downloading package knbc to /root/nltk_data...\n",
            "[nltk_data]    |   Package knbc is already up-to-date!\n",
            "[nltk_data]    | Downloading package lin_thesaurus to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package lin_thesaurus is already up-to-date!\n",
            "[nltk_data]    | Downloading package mac_morpho to /root/nltk_data...\n",
            "[nltk_data]    |   Package mac_morpho is already up-to-date!\n",
            "[nltk_data]    | Downloading package machado to /root/nltk_data...\n",
            "[nltk_data]    |   Package machado is already up-to-date!\n",
            "[nltk_data]    | Downloading package masc_tagged to /root/nltk_data...\n",
            "[nltk_data]    |   Package masc_tagged is already up-to-date!\n",
            "[nltk_data]    | Downloading package moses_sample to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package moses_sample is already up-to-date!\n",
            "[nltk_data]    | Downloading package movie_reviews to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package movie_reviews is already up-to-date!\n",
            "[nltk_data]    | Downloading package names to /root/nltk_data...\n",
            "[nltk_data]    |   Package names is already up-to-date!\n",
            "[nltk_data]    | Downloading package nombank.1.0 to /root/nltk_data...\n",
            "[nltk_data]    |   Package nombank.1.0 is already up-to-date!\n",
            "[nltk_data]    | Downloading package nps_chat to /root/nltk_data...\n",
            "[nltk_data]    |   Package nps_chat is already up-to-date!\n",
            "[nltk_data]    | Downloading package omw to /root/nltk_data...\n",
            "[nltk_data]    |   Package omw is already up-to-date!\n",
            "[nltk_data]    | Downloading package opinion_lexicon to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package opinion_lexicon is already up-to-date!\n",
            "[nltk_data]    | Downloading package paradigms to /root/nltk_data...\n",
            "[nltk_data]    |   Package paradigms is already up-to-date!\n",
            "[nltk_data]    | Downloading package pil to /root/nltk_data...\n",
            "[nltk_data]    |   Package pil is already up-to-date!\n",
            "[nltk_data]    | Downloading package pl196x to /root/nltk_data...\n",
            "[nltk_data]    |   Package pl196x is already up-to-date!\n",
            "[nltk_data]    | Downloading package ppattach to /root/nltk_data...\n",
            "[nltk_data]    |   Package ppattach is already up-to-date!\n",
            "[nltk_data]    | Downloading package problem_reports to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package problem_reports is already up-to-date!\n",
            "[nltk_data]    | Downloading package propbank to /root/nltk_data...\n",
            "[nltk_data]    |   Package propbank is already up-to-date!\n",
            "[nltk_data]    | Downloading package ptb to /root/nltk_data...\n",
            "[nltk_data]    |   Package ptb is already up-to-date!\n",
            "[nltk_data]    | Downloading package product_reviews_1 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package product_reviews_1 is already up-to-date!\n",
            "[nltk_data]    | Downloading package product_reviews_2 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package product_reviews_2 is already up-to-date!\n",
            "[nltk_data]    | Downloading package pros_cons to /root/nltk_data...\n",
            "[nltk_data]    |   Package pros_cons is already up-to-date!\n",
            "[nltk_data]    | Downloading package qc to /root/nltk_data...\n",
            "[nltk_data]    |   Package qc is already up-to-date!\n",
            "[nltk_data]    | Downloading package reuters to /root/nltk_data...\n",
            "[nltk_data]    |   Package reuters is already up-to-date!\n",
            "[nltk_data]    | Downloading package rte to /root/nltk_data...\n",
            "[nltk_data]    |   Package rte is already up-to-date!\n",
            "[nltk_data]    | Downloading package semcor to /root/nltk_data...\n",
            "[nltk_data]    |   Package semcor is already up-to-date!\n",
            "[nltk_data]    | Downloading package senseval to /root/nltk_data...\n",
            "[nltk_data]    |   Package senseval is already up-to-date!\n",
            "[nltk_data]    | Downloading package sentiwordnet to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package sentiwordnet is already up-to-date!\n",
            "[nltk_data]    | Downloading package sentence_polarity to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package sentence_polarity is already up-to-date!\n",
            "[nltk_data]    | Downloading package shakespeare to /root/nltk_data...\n",
            "[nltk_data]    |   Package shakespeare is already up-to-date!\n",
            "[nltk_data]    | Downloading package sinica_treebank to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package sinica_treebank is already up-to-date!\n",
            "[nltk_data]    | Downloading package smultron to /root/nltk_data...\n",
            "[nltk_data]    |   Package smultron is already up-to-date!\n",
            "[nltk_data]    | Downloading package state_union to /root/nltk_data...\n",
            "[nltk_data]    |   Package state_union is already up-to-date!\n",
            "[nltk_data]    | Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]    |   Package stopwords is already up-to-date!\n",
            "[nltk_data]    | Downloading package subjectivity to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package subjectivity is already up-to-date!\n",
            "[nltk_data]    | Downloading package swadesh to /root/nltk_data...\n",
            "[nltk_data]    |   Package swadesh is already up-to-date!\n",
            "[nltk_data]    | Downloading package switchboard to /root/nltk_data...\n",
            "[nltk_data]    |   Package switchboard is already up-to-date!\n",
            "[nltk_data]    | Downloading package timit to /root/nltk_data...\n",
            "[nltk_data]    |   Package timit is already up-to-date!\n",
            "[nltk_data]    | Downloading package toolbox to /root/nltk_data...\n",
            "[nltk_data]    |   Package toolbox is already up-to-date!\n",
            "[nltk_data]    | Downloading package treebank to /root/nltk_data...\n",
            "[nltk_data]    |   Package treebank is already up-to-date!\n",
            "[nltk_data]    | Downloading package twitter_samples to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package twitter_samples is already up-to-date!\n",
            "[nltk_data]    | Downloading package udhr to /root/nltk_data...\n",
            "[nltk_data]    |   Package udhr is already up-to-date!\n",
            "[nltk_data]    | Downloading package udhr2 to /root/nltk_data...\n",
            "[nltk_data]    |   Package udhr2 is already up-to-date!\n",
            "[nltk_data]    | Downloading package unicode_samples to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package unicode_samples is already up-to-date!\n",
            "[nltk_data]    | Downloading package universal_treebanks_v20 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package universal_treebanks_v20 is already up-to-\n",
            "[nltk_data]    |       date!\n",
            "[nltk_data]    | Downloading package verbnet to /root/nltk_data...\n",
            "[nltk_data]    |   Package verbnet is already up-to-date!\n",
            "[nltk_data]    | Downloading package verbnet3 to /root/nltk_data...\n",
            "[nltk_data]    |   Package verbnet3 is already up-to-date!\n",
            "[nltk_data]    | Downloading package webtext to /root/nltk_data...\n",
            "[nltk_data]    |   Package webtext is already up-to-date!\n",
            "[nltk_data]    | Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]    |   Package wordnet is already up-to-date!\n",
            "[nltk_data]    | Downloading package wordnet_ic to /root/nltk_data...\n",
            "[nltk_data]    |   Package wordnet_ic is already up-to-date!\n",
            "[nltk_data]    | Downloading package words to /root/nltk_data...\n",
            "[nltk_data]    |   Package words is already up-to-date!\n",
            "[nltk_data]    | Downloading package ycoe to /root/nltk_data...\n",
            "[nltk_data]    |   Package ycoe is already up-to-date!\n",
            "[nltk_data]    | Downloading package rslp to /root/nltk_data...\n",
            "[nltk_data]    |   Package rslp is already up-to-date!\n",
            "[nltk_data]    | Downloading package maxent_treebank_pos_tagger to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package maxent_treebank_pos_tagger is already up-\n",
            "[nltk_data]    |       to-date!\n",
            "[nltk_data]    | Downloading package universal_tagset to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package universal_tagset is already up-to-date!\n",
            "[nltk_data]    | Downloading package maxent_ne_chunker to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package maxent_ne_chunker is already up-to-date!\n",
            "[nltk_data]    | Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]    |   Package punkt is already up-to-date!\n",
            "[nltk_data]    | Downloading package book_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package book_grammars is already up-to-date!\n",
            "[nltk_data]    | Downloading package sample_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package sample_grammars is already up-to-date!\n",
            "[nltk_data]    | Downloading package spanish_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package spanish_grammars is already up-to-date!\n",
            "[nltk_data]    | Downloading package basque_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package basque_grammars is already up-to-date!\n",
            "[nltk_data]    | Downloading package large_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package large_grammars is already up-to-date!\n",
            "[nltk_data]    | Downloading package tagsets to /root/nltk_data...\n",
            "[nltk_data]    |   Package tagsets is already up-to-date!\n",
            "[nltk_data]    | Downloading package snowball_data to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package snowball_data is already up-to-date!\n",
            "[nltk_data]    | Downloading package bllip_wsj_no_aux to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package bllip_wsj_no_aux is already up-to-date!\n",
            "[nltk_data]    | Downloading package word2vec_sample to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package word2vec_sample is already up-to-date!\n",
            "[nltk_data]    | Downloading package panlex_swadesh to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package panlex_swadesh is already up-to-date!\n",
            "[nltk_data]    | Downloading package mte_teip5 to /root/nltk_data...\n",
            "[nltk_data]    |   Package mte_teip5 is already up-to-date!\n",
            "[nltk_data]    | Downloading package averaged_perceptron_tagger to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package averaged_perceptron_tagger is already up-\n",
            "[nltk_data]    |       to-date!\n",
            "[nltk_data]    | Downloading package averaged_perceptron_tagger_ru to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package averaged_perceptron_tagger_ru is already\n",
            "[nltk_data]    |       up-to-date!\n",
            "[nltk_data]    | Downloading package perluniprops to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package perluniprops is already up-to-date!\n",
            "[nltk_data]    | Downloading package nonbreaking_prefixes to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package nonbreaking_prefixes is already up-to-date!\n",
            "[nltk_data]    | Downloading package vader_lexicon to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Package vader_lexicon is already up-to-date!\n",
            "[nltk_data]    | Downloading package porter_test to /root/nltk_data...\n",
            "[nltk_data]    |   Package porter_test is already up-to-date!\n",
            "[nltk_data]    | Downloading package wmt15_eval to /root/nltk_data...\n",
            "[nltk_data]    |   Package wmt15_eval is already up-to-date!\n",
            "[nltk_data]    | Downloading package mwa_ppdb to /root/nltk_data...\n",
            "[nltk_data]    |   Package mwa_ppdb is already up-to-date!\n",
            "[nltk_data]    | \n",
            "[nltk_data]  Done downloading collection all\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-2d43a56d-0636-44fa-9a16-91d2da8d9812\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-2d43a56d-0636-44fa-9a16-91d2da8d9812\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving Test_set.xlsx to Test_set.xlsx\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'Test_set.xlsx': b'PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00b\\xee\\x9dh^\\x01\\x00\\x00\\x90\\x04\\x00\\x00\\x13\\x00\\x08\\x02[Content_Types].xml \\xa2\\x04\\x02(\\xa0\\x00\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xac\\x94\\xcbN\\xc30\\x10E\\xf7H\\xfcC\\xe4-J\\xdc\\xb2@\\x085\\xed\\x82\\xc7\\x12*Q>\\xc0\\xc4\\x93\\xc6\\xaac[\\x9eii\\xff\\x9e\\x89\\xfb\\x10B\\xa1\\x15j7\\xb1\\x12\\xcf\\xdc{2\\xf1\\xcdh\\xb2nm\\xb6\\x82\\x88\\xc6\\xbbR\\x0c\\x8b\\x81\\xc8\\xc0U^\\x1b7/\\xc5\\xc7\\xec%\\xbf\\x17\\x19\\x92rZY\\xef\\xa0\\x14\\x1b@1\\x19__\\x8df\\x9b\\x00\\x98q\\xb7\\xc3R4D\\xe1AJ\\xac\\x1ah\\x15\\x16>\\x80\\xe3\\x9d\\xda\\xc7V\\x11\\xdf\\xc6\\xb9\\x0c\\xaaZ\\xa89\\xc8\\xdb\\xc1\\xe0NV\\xde\\x118\\xca\\xa9\\xd3\\x10\\xe3\\xd1\\x13\\xd4ji){^\\xf3\\xe3-I\\x04\\x8b\"{\\xdc\\x16v^\\xa5P!XS)bR\\xb9r\\xfa\\x97K\\xbes(\\xb83\\xd5`c\\x02\\xde0\\x86\\x90\\xbd\\x0e\\xdd\\xce\\xdf\\x06\\xbb\\xbe7\\x1eM4\\x1a\\xb2\\xa9\\x8a\\xf4\\xaaZ\\xc6\\x90k+\\xbf|\\\\|z\\xbf(\\x8e\\x8b\\xf4P\\xfa\\xba6\\x15h_-[\\x9e@\\x81!\\x82\\xd2\\xd8\\x00Pk\\x8b\\xb4\\x16\\xad2n\\xcf}\\xc4?\\x15\\xa3L\\xcb\\xf0\\xc2 \\xdd\\xfb%\\xe1\\x13\\x1c\\xc4\\xdf\\x1bd\\xba\\x9e\\x8f\\x90dN\\x18\"m,\\xe0\\xa5\\xc7\\x9eDO97*\\x82~\\xa7\\xc8\\xc9\\xb88\\xc0O\\xedc\\x1c|n\\xa6\\xd1\\x07\\xe4\\x04E\\xf8\\xff\\x14\\xf6\\x11\\xe9\\xba\\xf3\\xc0B\\x10\\xc9\\xc0!$}\\x87\\xed\\xe0\\xc8\\xe9;{\\xec\\xd0\\xe5[\\x83\\xee\\xf1\\x96\\xe9\\x7f2\\xfe\\x06\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xb5U0#\\xf4\\x00\\x00\\x00L\\x02\\x00\\x00\\x0b\\x00\\x08\\x02_rels/.rels \\xa2\\x04\\x02(\\xa0\\x00\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xac\\x92MO\\xc30\\x0c\\x86\\xefH\\xfc\\x87\\xc8\\xf7\\xd5\\xdd\\x90\\x10BKwAH\\xbb!T~\\x80I\\xdc\\x0f\\xb5\\x8d\\xa3$\\x1b\\xdd\\xbf\\'\\x1c\\x10T\\x1a\\x83\\x03G\\x7f\\xbd~\\xfc\\xca\\xdb\\xdd<\\x8d\\xea\\xc8!\\xf6\\xe24\\xac\\x8b\\x12\\x14;#\\xb6w\\xad\\x86\\x97\\xfaqu\\x07*&r\\x96Fq\\xac\\xe1\\xc4\\x11v\\xd5\\xf5\\xd5\\xf6\\x99GJy(v\\xbd\\x8f*\\xab\\xb8\\xa8\\xa1K\\xc9\\xdf#F\\xd3\\xf1D\\xb1\\x10\\xcf.W\\x1a\\t\\x13\\xa5\\x1c\\x86\\x16=\\x99\\x81Z\\xc6MY\\xdeb\\xf8\\xae\\x01\\xd5BS\\xed\\xad\\x86\\xb0\\xb77\\xa0\\xea\\x93\\xcf\\x9b\\x7f\\xd7\\x96\\xa6\\xe9\\r?\\x889L\\xec\\xd2\\x99\\x15\\xc8sbg\\xd9\\xae|\\xc8l!\\xf5\\xf9\\x1aUSh9i\\xb0b\\x9er:\"y_dl\\xc0\\xf3D\\x9b\\xbf\\x13\\xfd|-N\\x9c\\xc8R\"4\\x12\\xf82\\xcfG\\xc7%\\xa0\\xf5\\x7fZ\\xb44\\xf1\\xcb\\x9dy\\xc47\\t\\xc3\\xab\\xc8\\xf0\\xc9\\x82\\x8b\\x1f\\xa8\\xde\\x01\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xbd\\x92e\\xe3W\\x03\\x00\\x00+\\x08\\x00\\x00\\x0f\\x00\\x00\\x00xl/workbook.xml\\xacUko\\xa38\\x14\\xfd\\xbe\\xd2\\xfe\\x07\\xc4w\\x8a\\xcd+\\x045\\x19\\x95\\x97\\xb6\\xd2tT\\xb5\\x9d\\xce\\x97J+\\x17L\\xb1\\x02\\x985\\xa6IU\\xcd\\x7f\\x9fk\\x13\\xd2v:Ze;\\x1b%6~p|\\x8e\\xef\\xb97\\xa7\\x9fvmc<R10\\xde\\xadL|\\x82L\\x83v\\x05/Y\\xf7\\xb02\\xbf\\xde\\xe4Vh\\x1a\\x83$]I\\x1a\\xde\\xd1\\x95\\xf9D\\x07\\xf3\\xd3\\xfa\\xcf?N\\xb7\\\\l\\xee9\\xdf\\x18\\x00\\xd0\\r+\\xb3\\x96\\xb2\\x8fl{(j\\xda\\x92\\xe1\\x84\\xf7\\xb4\\x83\\x95\\x8a\\x8b\\x96H\\x18\\x8a\\x07{\\xe8\\x05%\\xe5PS*\\xdb\\xc6v\\x10\\n\\xec\\x96\\xb0\\xce\\x9c\\x10\"q\\x0c\\x06\\xaf*V\\xd0\\x94\\x17cK;9\\x81\\x08\\xda\\x10\\t\\xf4\\x87\\x9a\\xf5\\xc3\\x8c\\xd6\\x16\\xc7\\xc0\\xb5Dl\\xc6\\xde*x\\xdb\\x03\\xc4=k\\x98|\\xd2\\xa0\\xa6\\xd1\\x16\\xd1\\xf9C\\xc7\\x05\\xb9o@\\xf6\\x0e\\xfb\\xc6N\\xc07\\x80\\x1fF\\xd08\\xf3I\\xb0\\xf4\\xee\\xa8\\x96\\x15\\x82\\x0f\\xbc\\x92\\'\\x00mO\\xa4\\xdf\\xe9\\xc7\\xc8\\xc6\\xf8\\xcd\\x15\\xec\\xde\\xdf\\xc1qH\\x9e-\\xe8#S1<\\xb0\\x12\\xc1\\x07Y\\x05\\x07\\xac\\xe0\\x05\\x0c\\xa3\\xdfF\\xc3`-\\xed\\x95\\x08.\\xef\\x83h\\xfe\\x81\\x9bc\\xaeO+\\xd6\\xd0\\xdb\\xc9\\xba\\x06\\xe9\\xfb/\\xa4U\\x91jL\\xa3!\\x83\\xccJ&i\\xb92\\x170\\xe4[\\xfafB\\x8c}<\\xb2\\x06V\\x1d\\x179K\\xd3^\\x1f\\xec|)\\x8c\\x92Vdl\\xe4\\r\\x18y\\x86\\x87\\xcc\\x08\\x82\\xa5\\xe3\\xab\\x9d`\\x8c\\xb3FR\\xd1\\x11I\\x13\\xdeI\\xf0\\xe1^\\xd7\\xefzNc\\'5\\x07\\x87\\x1bW\\xf4\\x9f\\x91\\t\\n\\x89\\x05\\xfe\\x02\\xad\\xd0\\x92\"\"\\xf7\\xc3%\\x91\\xb51\\x8afe&\\xd1\\xdd\\xd7\\x01\\xe4\\xdf\\x11\"xw\\x97\\xd2a#y\\x7f\\xf7\\xca\\x97\\xe4}\\x12\\xfc\\x07g\\x92B\\xc9\\xb5A\\xef\\xc4iz\\xfeY;P\\x13\\xd1\\xec\\xbeK)\\x0cx>O?C\\x04\\xae\\xc9#\\xc4\\x03\\xa2^\\xee\\xd3\\xf5\\x1c.<\\xfc\\xfb9\\r\\xd1\"\\x0c\\x82\\xc4\\xca\\xbc\\xdc\\xb7<\\x843+\\x8e\\x11\\xb2\\xc2|q\\x16\\xbb\\xd8\\x0b]\\xd7\\xff\\x0e*D\\x10\\x15\\x9c\\x8c\\xb2\\xde\\xc7Xa\\xaeL\\xcf\\xff\\xc5\\xd2\\x05\\xd9\\xcd+\\x18E#+_\\xce\\x7fF\\xfb\\x8f\\xa5\\xfa\\x9f\\x9ay\\xed\\xbbR\\xaa\\xaa\\xd9-\\xa3\\xdb\\xe1\\xc5\\rjh\\xec\\xbe\\xb1\\xae\\xe4[\\xf0\\xc0\\xc2\\x87\\xc3\\x9f\\x0e\\xc3\\xc0\\x07q[\\xbd\\xf8\\x8d\\x95\\xb2\\x06;\\xe1\\x00\\x1d\\xe6\\xfe\\xa2\\xec\\xa1\\x06\\xc6\\x18\\xbb\\xa1&\\xed(f+\\xf3\\xd9\\t\\xc3 \\xcd|\\x1fd/\\xb1\\xe5\\xf9\\x19hO\\xb2\\xd4\\n\\x9c`\\x91\\xc7\\xd8O|\\'\\xd0\\x8c\\xecW\\x94t\\xdd\\x04j\\xba7:\\xed\\xf5kUK1\\x14h\\xd5\\xab\\xdb\\x85g\\x11\\xa93\\xc4y\\x89u\\xf4\\xe6\\xd7\\n\\xd2\\x14\\xe0m\\xd5\\xe9\\x8dK\\xbc7>\\xdd\\xc9\\xcf\\x83\\\\\\x9fB\\x0f\\xb6b@\\x0f{\\xe8l\\x81\\x96\\x9e\\x852\\x17\\xe2\\x13.\\x1d+\\xf4\\\\\\xc7J\\xbc\\xd4\\xc9\\xfcE\\x96f\\xb1\\x8e\\x8f\\xaa\\xfb\\xd1\\xffQ\\xfd\\xb4\\xbb\\xa3\\xf9\\x0fE\\xb1\\xac\\x89\\x907\\x82\\x14\\x1b\\xf8\\x1b\\xba\\xa2UL\\x06p\\xd2$\\x08x\\xbe&\\x1b\\xfba\\x8c\\\\\\xa0\\xe8\\xe58\\xb7<\\xbcDp\\xab\\x81g\\xf9i\\xee\\xfa\\x0b\\x9c&\\x99\\x9f+3Md\\x95\\xfc\\xea\\x83\\xb5\\'\\xb4\\xf5\\xdb\\x94\\xc8\\x11\\xf2R\\xa5\\xa4\\x1eG\\xaa\\xcd\\xf7\\xb3\\x87\\xc9j\\x9a\\xd8\\xc7\\xe9M\\xd2EW\\xa9\\x8a\\xcc\\xfe\\xed\\x7f\\xdbx\\r\\xea\\x1bz\\xe4\\xe6\\xfc\\xf6\\xc8\\x8d\\xc9\\x97\\x8b\\x9b\\x0b\\xed\\x8d_\\n\\xb0\\xf5\\x05\\xabV\\xdb\\xc2\\x9e\\xc3\\xb2\\xfe\\x01\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\x81>\\x94\\x97\\xf3\\x00\\x00\\x00\\xba\\x02\\x00\\x00\\x1a\\x00\\x08\\x01xl/_rels/workbook.xml.rels \\xa2\\x04\\x01(\\xa0\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xacRMK\\xc40\\x10\\xbd\\x0b\\xfe\\x870w\\x9bv\\x15\\x11\\xd9t/\"\\xecU\\xeb\\x0f\\x08\\xc9\\xb4)\\xdb&!3~\\xf4\\xdf\\x1b*\\xba]X\\xd6K/\\x03o\\x86y\\xef\\xcd\\xc7v\\xf75\\x0e\\xe2\\x03\\x13\\xf5\\xc1+\\xa8\\x8a\\x12\\x04z\\x13l\\xef;\\x05o\\xcd\\xf3\\xcd\\x03\\x08b\\xed\\xad\\x1e\\x82G\\x05\\x13\\x12\\xec\\xea\\xeb\\xab\\xed\\x0b\\x0e\\x9as\\x13\\xb9>\\x92\\xc8,\\x9e\\x148\\xe6\\xf8(%\\x19\\x87\\xa3\\xa6\"D\\xf4\\xb9\\xd2\\x864j\\xce0u2js\\xd0\\x1d\\xcaMY\\xde\\xcb\\xb4\\xe4\\x80\\xfa\\x84S\\xec\\xad\\x82\\xb4\\xb7\\xb7 \\x9a)f\\xe5\\xff\\xb9C\\xdb\\xf6\\x06\\x9f\\x82y\\x1f\\xd1\\xf3\\x19\\tI<\\ry\\x00\\xd1\\xe8\\xd4!+\\xf8\\xc1E\\xf6\\x08\\xf2\\xbc\\xfcfMy\\xcek\\xc1\\xa3\\xfa\\x0c\\xe5\\x1c\\xabK\\x1e\\xaa5=|\\x86t \\x87\\xc8G\\x1f\\x7f)\\x92s\\xe5\\xa2\\x99\\xbbU\\xef\\xe1tB\\xfb\\xca)\\xbf\\xdb\\xf2,\\xcb\\xf4\\xeff\\xe4\\xc9\\xc7\\xd5\\xdf\\x00\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00WJ\\x84%r\\x06\\x00\\x00\\xfb&\\x00\\x00\\x18\\x00\\x00\\x00xl/worksheets/sheet1.xml\\x9c\\x9a\\xdbn\\xdbF\\x10\\x86\\xef\\x0b\\xf4\\x1d\\x04\\xde\\xdb\\xda\\xf3A\\x90\\x14\\xd8q\\x83\\x16h\\x81\\xa2\\xa7{\\x9a\\xa2l\"\\x92\\xa8\\x92\\xb4\\x9d\\xa0\\xe8\\xbbw8\\xaa\\xe5\\xd5Z\\x03h\\x02\\xc4\\x80\\xec\\xfc\\xfa9\\xdc\\xddof\\xb8\\xcb\\xf9\\x87/\\xdb\\xcd\\xe4\\xb9\\xee\\xfa\\xa6\\xdd-\\ny-\\x8aI\\xbd\\xab\\xdaU\\xb3{X\\x14\\x7f\\xfe\\xf1\\xe9*\\x14\\x93~(w\\xabr\\xd3\\xee\\xeaE\\xf1\\xb5\\xee\\x8b\\x0f\\xcb\\xef\\xbf\\x9b\\xbf\\xb4\\xdd\\xe7\\xfe\\xb1\\xae\\x87\\t8\\xec\\xfaE\\xf18\\x0c\\xfb\\xd9t\\xdaW\\x8f\\xf5\\xb6\\xec\\xaf\\xdb}\\xbd\\x83\\xffY\\xb7\\xdd\\xb6\\x1c\\xe0\\xd7\\xeea\\xda\\xef\\xbb\\xba\\\\\\xe1\\x97\\xb6\\x9b\\xa9\\x12\\xc2M\\xb7e\\xb3+\\x0e\\x0e\\xb3\\xee\\x12\\x8fv\\xbdn\\xaa\\xfa\\xae\\xad\\x9e\\xb6\\xf5n8\\x98t\\xf5\\xa6\\x1c \\xfe\\xfe\\xb1\\xd9\\xf7\\xafn\\xdb\\xea\\x12\\xbbm\\xd9}~\\xda_U\\xedv\\x0f\\x16\\xf7\\xcd\\xa6\\x19\\xbe\\xa2i1\\xd9V\\xb3\\x9f\\x1evmW\\xdeo\\xe0\\xbe\\xbfHSV\\x93/\\x1d\\xfcS\\xf0\\xa3_/\\x83\\x7f\\x7fw\\xa5mSum\\xdf\\xae\\x87kp\\x9e\\x1eb~\\x7f\\xfbq\\x1a\\xa7eutz\\x7f\\xff\\x17\\xd9H3\\xed\\xea\\xe7f\\x9c\\xc07+\\xf5m!I{\\xf4Rof\\xfa\\x1b\\xcd\\xdc\\xd1l\\x1c\\xaen\\xf6\\xd4\\xac\\x16\\xc5?B\\xfa(\\xe4\\xdd\\xed\\xd5\\x8d\\x8a?\\\\\\x99x\\x03kL\\xf8\\xdb\\xab\\x8f\\x9fn\\xd5\\xdd\\x8d\\xb6\\xf1\\xe6V\\xff[,\\xe7\\xab\\x06fx\\xbc\\xabIW\\xaf\\x17\\xc5\\x8d\\x9c\\xddH!\\x8b\\xe9r\\x8e+\\xe8\\xaf\\xa6~\\xe9\\x93\\xcf\\x93\\xa1\\xbc\\xff\\xbd\\xde\\xd4\\xd5P\\xc3Ud1\\x19\\xda\\xfd\\xcf\\xf5z\\xf8Xo6\\xf0\\xed\\x00\\xebx\\\\\\xb2\\xf7m\\xfby\\xfc\\xeaO \\x12p\\x95\\x1e\\xbf2^\\xa5\\xff;\\xbf\\xce\\xf4x\\xa1\\xe5\\xfc\\xed\\xf3\\xebE?\\xe1\\xca\\xfe\\xb5\\x9b\\xac\\xeau\\xf9\\xb4\\x19~k_~\\xac\\x9b\\x87\\xc7\\x01\\xaen\\xe1~\\xc7\\x053[}\\xbd\\xab\\xfb\\nV*\\\\\\xedZ\\xd9c\\xf4w\\xe5P.\\xe7]\\xfb2\\x81Y\\x87`\\xfb}92$g\\xf0\\xf9\\xfc7\\x97\\xf3j\\xd4\\xde\\x8c\\xe2\\xc3\\xfd-\\x8a\\x1en\\xe0y)\\xe6\\xd3g\\x08\\xb0\\x82\\x1f0<\\xba\\xc2\\x04^\\xee:\\x8aG\\xd7\\xd1O\\xc3\\x0c\\xe9h\\t[\\x98\\xca\\xcbmGqj\\x1b\\xa5\\xd2\\xe7\\xa35\\x1c\\xdbQ\\x9c\\xd8\\x1a\\x1f\\ra\\x0b\\x13qy\\xb4\\xa38\\xb1\\xb5\\xdeF\\x7f>Z\\xc7\\xb1\\x1d\\xc5\\x89\\xad\\xf22\\x10\\xd1z\\x8e\\xed(N\\x07\\xc1j\\xef\\xceG;\\xa6\\xf0\\x8b\\xd7\\xd7(NlCT\\xd1\\x9c\\xb7\\x8d\\x1c\\xdbQ\\x9c\\xd8F\\xeb\\xa3<o+\\xa1\\x041p\\x18\\xd5\\x89\\xb1\\x0b\\x96Z\\xb9\\x92\\xc7\\xd9+h\\xff#a\\x8c6\\x96\\x88\\x98\\x85\\x9a\\xccY\\xf3\\x81\\x1c\\n\\x16l2\\xa7\\xcd\\xd8\\x10\\x88\\x88Y\\xb8\\xc9\\x8c7\\xafm \\xb2\\xc3\\x98\\xf9\\x18\\x93\\x97\\x11g\\x94w\\x04\\x1a\\x92\\x85\\x1c\\xaaOV\\xb10\\xc4*\\x96,\\xe8P\\x9dR\\x17\\xa4\\xa3\\x8cY\\xd8\\xc9\\x8c;\\x1f\\xac\\xa7&\\x8f\\x05\\x9e\\xcc\\xc8S\\xdeXb\\x1d+\\x16y\\xa8N\\xf3\\x9a\\x95\\x86\\x88X\\xb1\\xc8Cu\\x9a+\\x8c\\t\\x8a\\xa8q\\xbc\"\\x97\\x91g\\xac\\x8b\\x941\\x8b<\\x95\\x91g\\x8c\\xd5TUf\\x91\\xa72\\xf2\\\\\\x10\\x14\\xd2\\xd0\\\\p\\xea}F\\x9ew\\xde\\x12\\xd5C\\xb1\\xc8C\\xf5I\\xc9\\xd7\\x8a\\x1a\\n\\x16y*\\xabw\\xce\\x1a\\x1d\\x89U\\xc1\"O\\xe5\\x15\\xcf\\x9bHE\\xcc\"O\\xe5\\xe4\\t/\\x881\\xd6,\\xf2P\\x9d\\x8e\\xb1\\x83\\x8c|~(4\\x8b<T\\x9f\\xb6*\\x96\\xa8\\xd2\\x9aE\\x1e\\xaa\\xd3*\\xedT\\xa4\"\\xe65\\x98\\x19y*HK$!\\xcd\"\\x0f\\xd5I\\xc4R9O$z\\xcd\"\\x0f\\xd5\\xe9\\x18K\\xa9(c\\x16y:\\xeb3\\xbd\\x97\\x14 \\x9aE\\x1e\\xaa\\xd3\\x9a\\x17}$\\xc8\\xd3,\\xf2P\\x9d\\x18kX\\xc7D>\\xd6,\\xf2P\\x9d\\x02\"\\x8c$\\xc6\\xd8\\xb0\\xc8C\\xf5\\ty\\xce\\x11\\x80\\x18\\x16y\\xa8N\\x1b\\x16\\x15\\xa8\\x86\\xc5\\xb0\\xc8Cu6\\xc6\\xd4#\\x18\\x8b<\\x93\\x91\\x17\\x84\\xa4\\xdaX\\xc3{\\xb8\\xcbj\\x9e\\x0ed\\xc3bX\\xe4\\xa1:\\x19\\n\\x1b\\x03\\x95+\\x0c\\x8b<T\\xa7c\\xac\\xbd\\xa0V\\x05\\x8b<\\x93\\xd5<xtt\\xc4\\x13\\xa9a\\x91\\x87\\xea\\x93*-\\xa9\\'h\\xc3\"\\x0f\\xd5\\xe9\\x18\\x07g\\x89\\x88-\\x8b<T\\'\\xc6\\xc2\\x07M$z\\xcb\"\\x0f\\xd5\\'\\xd9MP\\xa5\\xc9\\xb2\\xc8Cu:\\x14\\xd0\\xd0\\x13i\\xd3\\xb2\\xc8CuZL\\xa3\\xa16\\x14,\\x8b<T\\xa75OjK\\xa4M\\xcb\\xdbX\\xc9\\xbaM(\\xd2\\x91\\xe8\\x84,\\x8b<T\\'\\x11G\\xa7,QA,\\x8b<T\\xa7\\x93\\xa7\\x8c\\xa0\\xb6\\x98X\\xe4\\xd9\\xbc\\xdbTd1\\xb5,\\xf2P\\x9dD\\xac oR\\xbbW,\\xf2\\\\\\xb6\\xc3b\\x9d\\xa4:z\\xc7\"\\x0f\\xd5\\xe9\\x18K\\xed\\x08\\xa4\\x1d\\x8b<T\\xa7\\x8088k 6\\xf2X\\xe4\\xb9\\xbc\\xdb\\xd4F\\x10\\x808\\x16y\\xa8N\\x87\\xc2\\x93H;\\x16y\\xa8N\\x13\\xbd\\xd3\\x14 \\x8e\\xb7\\xa9\\x99w\\x9b&\\x04\\xa2\\xa3w,\\xf2P\\x9dD<V\\x7fj\\xf2X\\xe4\\xb9S\\xf2 \\xcb;IE\\xcc\"\\xcfe\\xcfy\\xce\\x92\\xfb\\xc6\\x9eE\\x1e\\xaaS\\xa4\\xbd\\xd5D\\xda\\xf4,\\xf2P\\x9d\\x8e\\xb1W\\x81\\xc8\\x15\\x9eE\\x1e\\xaa\\xd3\\xe5&\\x0c\\xb5\\xa1\\xe7Y\\xe4\\xa1:E\\x1a6,\\x88b\\xeaY\\xe4\\xa1:1\\xd6Vj\\x02i\\xcf\"\\x0f\\xd5ii\\x82\\x0e\\x8bh\\n=\\x8b<T\\x9f\\xa4MCu\\x9b\\x9ew\\xa0\\x90u\\x9b\\xc1;G\\x00\\xe2Y\\xe4\\xa1\\xfa-b\\x11\\r\\xf4\\xb1\\xe7\\xf3\\xb1g\\x91\\x87\\xead(\\x8c\\n\\x14 \\x81E\\x1e\\xaa\\x13c\\xe1\\x9d!\\x96[`\\x91\\x87\\xea\\x14\\x10/\\xa8\\xbd\\xcd\\xc0\"\\x0f\\xd5\\xa91\\xe4 \\xa2\\x98\\x06\\x16y\\xa8Ns\\x05\\x1c\\xdf\\x12\\xab\"\\xb0\\xc8Cu\\xba\\x8e\\xb5\\xa6\\xf6\\xdd\\x02\\x8b<T\\xa7\\xb9\\xc2Z\\xeaY:\\xb0\\xc8Cu\\x1aq ;\\xa1\\xc0\"\\x0f\\xd5\\xe9\\xe4iG\\xa5\\xcd\\xf1\\x1c\\x9bq\\x98\\xf7noS\\x91c\\xcc\"/\\xe4{\\x9bAEb\\xb9E\\x16y\\xa8N\\x86BI/\\x89\\x8e>\\xb2\\xc8C\\xf5I\\xae\\x10\\x822f\\x91\\x17\\xf3\\xf3<a\\xa9\\x83\\xc2\\xc8\"\\x0f\\xd5\\xe9\\xaa0R\\x12I(\\xb2\\xc8C\\xf5I\\x12R\\xd4\\xb1|d\\x91\\x87\\xea\\x94\\xbc\\x18\\xa8\\x13\\xf4\\xc8\"\\x0f\\xd5\\'5OE\\xe2q,\\xb2\\xc8Cu:\\x14\\nJ\\x08q\\xdc\\xcd\"/f\\xe4\\x19#\\xc8\\xa1\\xe0\\x1d\\xa4g\\xe4\\xf9\\xa0$y\\x92\\xce<J\\xcf\\x9e\\xf44\\x9c\\xc7\\x12m\\xe1\\xf8&\\x0e\\xeb\\x94>;M\\xb7\\xc1\\xaa\\x9c\\xbe\\xc3+6\\x877c\\xf6\\xe5C\\xfdK\\xd9=4\\xbb~\\xb2\\x81\\xb7x\\xc6\\xd7g`b\\xbb\\xc3\\xfb5\\xf8\\x19\\xde\\xef\\xc1\\xbf\\xc2\\n\\xbdo\\x87\\xa1\\xdd\\xbe\\xfe\\xf6\\x08/\\x99\\xd5\\xf0\\xaa\\x8c\\xb8\\x06\\xdc\\xd6m;\\xbc\\xfe\\x02\\xaf\\x0eM\\x8f\\xaf\\xad-\\xff\\x03\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xc1\\x17\\x10\\xbeN\\x07\\x00\\x00\\xc6 \\x00\\x00\\x13\\x00\\x00\\x00xl/theme/theme1.xml\\xecY\\xcd\\x8b\\x1b7\\x14\\xbf\\x17\\xfa?\\x0csw\\xfc5\\xe3\\x8f%\\xde\\xe0\\xcfl\\x93\\xdd$d\\x9d\\x94\\x1c\\xb5\\xb6\\xecQV32\\x92\\xbc\\x1b\\x13\\x02%9\\xf5R(\\xa4\\xa5\\x97Bo=\\x94\\xd2@\\x03\\r\\xbd\\xf4\\x8f\\t$\\xb4\\xe9\\x1f\\xd1\\'\\xcd\\xd8#\\xad\\xe5$\\x9blJZv\\r\\x8bG\\xfe\\xbd\\xa7\\xa7\\xf7\\x9e~z\\xf3t\\xf1\\xd2\\xbd\\x98zG\\x98\\x0b\\xc2\\x92\\x96_\\xbeP\\xf2=\\x9c\\x8c\\xd8\\x98$\\xd3\\x96\\x7fk8(4|OH\\x94\\x8c\\x11e\\tn\\xf9\\x0b,\\xfcK\\xdb\\x9f~r\\x11m\\xc9\\x08\\xc7\\xd8\\x03\\xf9Dl\\xa1\\x96\\x1fI9\\xdb*\\x16\\xc5\\x08\\x86\\x91\\xb8\\xc0f8\\x81\\xdf&\\x8c\\xc7H\\xc2#\\x9f\\x16\\xc7\\x1c\\x1d\\x83\\xde\\x98\\x16+\\xa5R\\xad\\x18#\\x92\\xf8^\\x82bP{}2!#\\xec\\r\\x95J\\x7f{\\xa9\\xbcO\\xe11\\x91B\\r\\x8c(\\xdfW\\xaa\\xb1%\\xa1\\xb1\\xe3\\xc3\\xb2B\\x88\\x85\\xe8R\\xee\\x1d!\\xda\\xf2a\\x9e1;\\x1e\\xe2{\\xd2\\xf7(\\x12\\x12~h\\xf9%\\xfd\\xe7\\x17\\xb7/\\x16\\xd1V&D\\xe5\\x06YCn\\xa0\\xff2\\xb9L`|X\\xd1s\\xf2\\xe9\\xc1j\\xd2 \\x08\\x83Z{\\xa5_\\x03\\xa8\\\\\\xc7\\xf5\\xeb\\xfdZ\\xbf\\xb6\\xd2\\xa7\\x01h4\\x82\\x95\\xa6\\xb6\\xd8:\\xeb\\x95n\\x90a\\rP\\xfa\\xd5\\xa1\\xbbW\\xefU\\xcb\\x16\\xde\\xd0_]\\xb3\\xb9\\x1d\\xaa\\x8f\\x85\\xd7\\xa0T\\x7f\\xb0\\x86\\x1f\\x0c\\xba\\xe0E\\x0b\\xafA)>\\\\\\xc3\\x87\\x9df\\xa7g\\xeb\\xd7\\xa0\\x14_[\\xc3\\xd7K\\xed^P\\xb7\\xf4kPDIr\\xb8\\x86.\\x85\\xb5jw\\xb9\\xda\\x15d\\xc2\\xe8\\x8e\\x13\\xde\\x0c\\x83A\\xbd\\x92)\\xcfQ\\x90\\r\\xab\\xecRSLX\"7\\xe5Z\\x8c\\xee2>\\x00\\x80\\x02R$I\\xe2\\xc9\\xc5\\x0cO\\xd0\\x08\\xb2\\xb8\\x8b(9\\xe0\\xc4\\xdb%\\xd3\\x08\\x12o\\x86\\x12&`\\xb8T)\\rJU\\xf8\\xaf>\\x81\\xfe\\xa6#\\x8a\\xb602\\xa4\\x95]`\\x89X\\x1bR\\xf6xb\\xc4\\xc9L\\xb6\\xfc+\\xa0\\xd57 /\\x9e={\\xfe\\xf0\\xe9\\xf3\\x87\\xbf=\\x7f\\xf4\\xe8\\xf9\\xc3_\\xb2\\xb9\\xb5*Kn\\x07%SS\\xee\\xd5\\x8f_\\xff\\xfd\\xfd\\x17\\xde_\\xbf\\xfe\\xf0\\xea\\xf17\\xe9\\xd4\\'\\xf1\\xc2\\xc4\\xbf\\xfc\\xf9\\xcb\\x97\\xbf\\xff\\xf1:\\xf5\\xb0\\xe2\\xdc\\x15/\\xbe}\\xf2\\xf2\\xe9\\x93\\x17\\xdf}\\xf5\\xe7O\\x8f\\x1d\\xda\\xdb\\x1c\\x1d\\x98\\xf0!\\x89\\xb1\\xf0\\xae\\xe1c\\xef&\\x8ba\\x81\\x0e\\xfb\\xf1\\x01?\\x9d\\xc40B\\xc4\\x92@\\x11\\xe8v\\xa8\\xee\\xcb\\xc8\\x02^[ \\xea\\xc2u\\xb0\\xed\\xc2\\xdb\\x1cX\\xc6\\x05\\xbc<\\xbfk\\xd9\\xba\\x1f\\xf1\\xb9$\\x8e\\x99\\xafF\\xb1\\x05\\xdcc\\x8cv\\x18w:\\xe0\\xaa\\x9a\\xcb\\xf0\\xf0p\\x9eL\\xdd\\x93\\xf3\\xb9\\x89\\xbb\\x89\\xd0\\x91k\\xee.J\\xac\\x00\\xf7\\xe73\\xa0W\\xe2R\\xd9\\x8d\\xb0e\\xe6\\r\\x8a\\x12\\x89\\xa68\\xc1\\xd2S\\xbf\\xb1C\\x8c\\x1d\\xab\\xbbC\\x88\\xe5\\xd7=2\\xe2L\\xb0\\x89\\xf4\\xee\\x10\\xaf\\x83\\x88\\xd3%Cr`%R.\\xb4Cb\\x88\\xcb\\xc2e \\x84\\xda\\xf2\\xcd\\xdem\\xaf\\xc3\\xa8k\\xd5=|d#a[ \\xea0~\\x88\\xa9\\xe5\\xc6\\xcbh.Q\\xecR9D15\\x1d\\xbe\\x8bd\\xe42r\\x7f\\xc1G&\\xae/$Dz\\x8a)\\xf3\\xfac,\\x84K\\xe6:\\x87\\xf5\\x1aA\\xbf\\n\\x0c\\xe3\\x0e\\xfb\\x1e]\\xc46\\x92Kr\\xe8\\xd2\\xb9\\x8b\\x183\\x91=v\\xd8\\x8dP<s\\xdaL\\x92\\xc8\\xc4~&\\x0e!E\\x91w\\x83I\\x17|\\x8f\\xd9;D=C\\x1cP\\xb21\\xdc\\xb7\\t\\xb6\\xc2\\xfdf\"\\xb8\\x05\\xe4j\\x9a\\x94\\'\\x88\\xfae\\xce\\x1d\\xb1\\xbc\\x8c\\x99\\xbd\\x1f\\x17t\\x82\\xb0\\x8be\\xda<\\xb6\\xd8\\xb5\\xcd\\x893;:\\xf3\\xa9\\x95\\xda\\xbb\\x18St\\x8c\\xc6\\x18{\\xb7>sX\\xd0a3\\xcb\\xe7\\xb9\\xd1W\"`\\x95\\x1d\\xecJ\\xac+\\xc8\\xceU\\xf5\\x9c`\\x01e\\x92\\xaak\\xd6)r\\x97\\x08+e\\xf7\\xf1\\x94m\\xb0goq\\x82x\\x16(\\x89\\x11\\xdf\\xa4\\xf9\\x1aD\\xddJ]8\\xe5\\x9cTz\\x9d\\x8e\\x0eM\\xe05\\x02\\xe5\\x1f\\xe4\\x8b\\xd3)\\xd7\\x05\\xe80\\x92\\xbb\\xbfI\\xeb\\x8d\\x08Yg\\x97z\\x16\\xee|]p+~o\\xb3\\xc7`_\\xde=\\xed\\xbe\\x04\\x19|j\\x19 \\xf6\\xb7\\xf6\\xcd\\x10Qk\\x82<a\\x86\\x08\\n\\x0c\\x17\\xdd\\x82\\x88\\x15\\xfe\\\\D\\x9d\\xabZl\\xee\\x94\\x9b\\xd8\\x9b6\\x0f\\x03\\x14FV\\xbd\\x13\\x93\\xe4\\x8d\\xc5\\xcf\\x89\\xb2\\'\\xfcw\\xca\\x1ew\\x01s\\x06\\x05\\x8f[\\xf1\\xfb\\x94:\\x9b(e\\xe7D\\x81\\xb3\\t\\xf7\\x1f,kzh\\x9e\\xdc\\xc0p\\x92\\xacs\\xd6yUs^\\xd5\\xf8\\xff\\xfb\\xaaf\\xd3^>\\xafe\\xcek\\x99\\xf3Z\\xc6\\xf5\\xf6\\xf5Aj\\x99\\xbc|\\x81\\xca&\\xef\\xf2\\xe8\\x9eO\\xbc\\xb1\\xe53!\\x94\\xee\\xcb\\x05\\xc5\\xbbBw}\\x04\\xbc\\xd1\\x8c\\x070\\xa8\\xdbQ\\xba\\'\\xb9j\\x01\\xce\"\\xf8\\x9a5\\x98,\\xdc\\x94#-\\xe3q&?\\'2\\xda\\x8f\\xd0\\x0cZCe\\xdd\\xc0\\x9c\\x8aL\\xf5Tx3&\\xa0c\\xa4\\x87u+\\x15\\x9f\\xd0\\xad\\xfbN\\xf3x\\x8f\\x8d\\xd3Ng\\xb9\\xac\\xba\\x9a\\xa9\\x0b\\x05\\x92\\xf9x)\\\\\\x8dC\\x97J\\xa6\\xe8Z=\\xef\\xde\\xad\\xd4\\xeb~\\xe8TwY\\x97\\x06(\\xd9\\xd3\\x18aLf\\x1bQu\\x18Q_\\x0eB\\x14^g\\x84^\\xd9\\x99X\\xd1tX\\xd1P\\xea\\x97\\xa1ZFq\\xe5\\n0m\\x15\\x15x\\xe5\\xf6\\xe0E\\xbd\\xe5\\x87A\\xdaA\\x86f\\x1c\\x94\\xe7c\\x15\\xa7\\xb4\\x99\\xbc\\x8c\\xae\\n\\xce\\x99Fz\\x933\\xa9\\x99\\x01Pb/3 \\x8ftS\\xd9\\xbaqyjui\\xaa\\xbdE\\xa4-#\\x8ct\\xb3\\x8d0\\xd20\\x82\\x17\\xe1,;\\xcd\\x96\\xfbY\\xc6\\xba\\x99\\x87\\xd42O\\xb9b\\xb9\\x1br3\\xea\\x8d\\x0f\\x11kE\"\\'\\xb8\\x81&&S\\xd0\\xc4;n\\xf9\\xb5j\\x08\\xb7*#4k\\xf9\\x13\\xe8\\x18\\xc3\\xd7x\\x06\\xb9#\\xd4[\\x17\\xa2S\\xb8v\\x19I\\x9en\\xf8wa\\x96\\x19\\x17\\xb2\\x87D\\x94:\\\\\\x93N\\xca\\x061\\x91\\x98{\\x94\\xc4-_-\\x7f\\x95\\r4\\xd1\\x1c\\xa2m+W\\x80\\x10>Z\\xe3\\x9a@+\\x1f\\x9bq\\x10t;\\xc8x2\\xc1#i\\x86\\xdd\\x18Q\\x9eN\\x1f\\x81\\xe1S\\xaep\\xfe\\xaa\\xc5\\xdf\\x1d\\xac$\\xd9\\x1c\\xc2\\xbd\\x1f\\x8d\\x8f\\xbd\\x03:\\xe77\\x11\\xa4XX/+\\x07\\x8e\\x89\\x80\\x8b\\x83r\\xea\\xcd1\\x81\\x9b\\xb0\\x15\\x91\\xe5\\xf9w\\xe2`\\xcah\\xd7\\xbc\\x8a\\xd29\\x94\\x8e#:\\x8bPv\\xa2\\x98d\\x9e\\xc25\\x89\\xae\\xcc\\xd1O+\\x1f\\x18O\\xd9\\x9a\\xc1\\xa1\\xeb.<\\x98\\xaa\\x03\\xf6\\xbdO\\xdd7\\x1f\\xd5\\xcas\\x06i\\xe6g\\xa6\\xc5*\\xea\\xd4t\\x93\\xe9\\x87;\\xe4\\r\\xab\\xf2C\\xd4\\xb2*\\xa5n\\xfdN-r\\xaek.\\xb9\\x0e\\x12\\xd5yJ\\xbc\\xe1\\xd4}\\x8b\\x03\\xc10-\\x9f\\xcc2MY\\xbcN\\xc3\\x8a\\xb3\\xb3Q\\xdb\\xb43,\\x08\\x0cO\\xd46\\xf8muF8=\\xf1\\xae\\'?\\xc8\\x9d\\xccZu@,\\xebJ\\x9d\\xf8\\xfa\\xca\\xdc\\xbc\\xd5f\\x07w\\x81<zp\\x7f8\\xa7R\\xe8PBo\\x97#(\\xfa\\xd2\\x1b\\xc8\\x946`\\x8b\\xdc\\x93Y\\x8d\\x08\\xdf\\xbc9\\'-\\xff~)l\\x07\\xddJ\\xd8-\\x94\\x1aa\\xbf\\x10T\\x83R\\xa1\\x11\\xb6\\xab\\x85v\\x18V\\xcb\\xfd\\xb0\\\\\\xeau*\\x0f\\xe0`\\x91Q\\\\\\x0e\\xd3\\xeb\\xfa\\x01\\\\a\\xd0Evi\\xaf\\xc7\\xd7.\\xee\\xe3\\xe5-\\xcd\\x85\\x11\\x8b\\x8bL_\\xcc\\x17\\xb5\\xe1\\xfa\\xe2\\xbe\\\\\\xd9|q\\xef\\x11 \\x9d\\xfb\\xb5\\xca\\xa0Ymvj\\x85f\\xb5=(\\x04\\xbdN\\xa3\\xd0\\xec\\xd6:\\x85^\\xad[\\xef\\rz\\xdd\\xb0\\xd1\\x1c<\\xf0\\xbd#\\r\\x0e\\xda\\xd5nP\\xeb7\\n\\xb5r\\xb7[\\x08j%e~\\xa3Y\\xa8\\x07\\x95J;\\xa8\\xb7\\x1b\\xfd\\xa0\\xfd +c`\\xe5)}d\\xbe\\x00\\xf7j\\xbb\\xb6\\xff\\x01\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xc9=c\\xbf\\xd1\\x02\\x00\\x00\\x0b\\x07\\x00\\x00\\r\\x00\\x00\\x00xl/styles.xml\\xa4Umo\\xda0\\x10\\xfe>i\\xff\\xc1\\xf2\\xf7\\xd4IJ\\x18\\xa0$\\xd5(\\x8dTi\\x9b&\\xb5\\x93\\xf6\\xd5$\\x0eX\\xf5Kd\\x1b\\x166\\xed\\xbf\\xef\\x9c\\x04H\\xd5\\xee\\xad\\xe5\\x03\\xd8g\\xdfs\\xcf=w>\\xd2\\xabV\\n\\xb4g\\xc6r\\xad2\\x1c]\\x84\\x181U\\xea\\x8a\\xabM\\x86\\xbf\\xdc\\x17\\xc1\\x0c#\\xeb\\xa8\\xaa\\xa8\\xd0\\x8ae\\xf8\\xc0,\\xbe\\xca\\xdf\\xbeI\\xad;\\x08v\\xb7e\\xcc!\\x80P6\\xc3[\\xe7\\x9a\\x05!\\xb6\\xdc2I\\xed\\x85n\\x98\\x82\\x93Z\\x1bI\\x1dl\\xcd\\x86\\xd8\\xc60ZY\\xef$\\x05\\x89\\xc3pJ$\\xe5\\n\\xf7\\x08\\x0bY\\xfe\\x0b\\x88\\xa4\\xe6a\\xd7\\x04\\xa5\\x96\\ru|\\xcd\\x05w\\x87\\x0e\\x0b#Y.n7J\\x1b\\xba\\x16@\\xb5\\x8d&\\xb4Dm451j\\xcd1Hg}\\x12G\\xf2\\xd2h\\xabkw\\x01\\xb8D\\xd75/\\xd9S\\xbas2\\'\\xb4<#\\x01\\xf2\\xcb\\x90\\xa2\\x84\\x84\\xf1\\xa3\\xdc[\\xf3B\\xa4\\t1l\\xcf}\\xf9p\\x9e\\xd6Z9\\x8bJ\\xbdS.\\xc31\\x10\\xf5\\x12,\\x1e\\x94\\xfe\\xa6\\n\\x7f\\x04\\x15\\x1en\\xe5\\xa9\\xfd\\x8e\\xf6T\\x80%\\xc2$OK-\\xb4A\\x0eJ\\x07\\xcau\\x16E%\\xebo\\\\S\\xc1\\xd7\\x86\\xfbk5\\x95\\\\\\x1czs\\xec\\r]\\xb5\\x87{\\x92\\x83\\xf6\\xdeH<\\x8f\\x9e\\xcd\\xf3q\\xccf\\x9d\\xe1\\xa2\\x08\\xbb\\x8f\\xf7x}\\xb0.\\xa6\\x85\\xa0\\\\\\x88\\x91\\x04\\xbd!O\\xa1W\\x1c3\\xaa\\x80S4\\xac\\xef\\x0f\\r\\xe4\\xaa\\xa0\\xad{\\xcep\\xf4\\xd7\\xdb\\x1bC\\x0fQ\\x9c\\x8c\\x1cH\\x170O\\xd7\\xdaT\\xf0\\x8c\\x8e\\xe2{\\x9d{S\\x9e\\nV;\\xc8\\xd1\\xf0\\xcd\\xd6\\xff:\\xdd\\xc0\\xf7Z;\\x07\\xad\\x96\\xa7\\x15\\xa7\\x1b\\xad\\xa8\\xf0\\xba\\x1d=\\x86\\x05\\xa4S2!\\xee\\xfcS\\xfbZ?\\xc2nk\\xa4v\\xb2\\x90\\xee\\xb6\\xca0<Z\\xaf\\xf8q\\t\\x89\\x0c\\xcb\\x1e\\xaf\\xdfx\\xfc1Z\\x8f=\\x82\\x8d\\x81\\xf2\\xff\\xc3\\xa2\\xb6>\\xe1\\xff\\xce;\\x02~\\xcf\\x93:y#\\xda4\\xe2\\xe0\\x9bth\\xbf\\x8e+\\xb0\\x1bI\\xf0H\\x80S*\\xc8\\xf7N\\x86?\\xf99#\\xa0\\xe5\\x07:h\\xbd\\xe3\\xc2q\\xf5L\\xf2\\x80Y\\xb5g9C_M\\xe7gF\\'\\xf4)\\n\\xa8Z\\xb1\\x9a\\xee\\x84\\xbb?\\x1df\\xf8\\xbc\\xfe\\xc8*\\xbe\\x93\\xf0\\xca\\x86[\\x9f\\xf9^\\xbb\\x0e\"\\xc3\\xe7\\xf5\\x07_\\xf5h\\xeac\\xb0\\xd6}\\xb0\\xf0.\\xe0\\x17\\xed\\x0c\\xcf\\xf0\\x8f\\x9b\\xe5\\xbb\\xf9\\xea\\xa6\\x88\\x83Y\\xb8\\x9c\\x05\\x93K\\x96\\x04\\xf3d\\xb9\\n\\x92\\xc9\\xf5r\\xb5*\\xe6a\\x1c^\\xff\\x1cM\\xaeW\\xcc\\xadn\\xd0B\\x81\\xa3\\xc9\\xc2\\n\\x98nfHv \\x7fw\\xb6ex\\xb4\\xe9\\xe9w\\xfd\\x0e\\xb4\\xc7\\xdc\\xe7\\xf14|\\x9fDaP\\\\\\x86Q0\\x99\\xd2Y0\\x9b^&A\\x91D\\xf1j:Y\\xde$E2\\xe2\\x9e\\xbcp\\xbe\\x85$\\x8a\\xfaI\\xe9\\xc9\\'\\x0b\\xc7%\\x13\\\\\\x1dku\\xac\\xd0\\xd8\\nE\\x82\\xed\\x1f\\x92 \\xc7J\\x90\\xf3\\xbfX\\xfe\\x0b\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\x1c&\\xa3z\\x9b\\x00\\x00\\x00\\xb4\\x00\\x00\\x00\\x14\\x00\\x00\\x00xl/sharedStrings.xml4\\xcdA\\n\\xc20\\x10\\x85\\xe1\\xbd\\xe0\\x1d\\xc2\\xecm\\xaa\\x0b\\x11I\\xd2\\x85\\xe0\\t\\xf4\\x00\\xa1\\x19\\xdb@3\\x89\\x99\\xa9\\xe8\\xed\\x8d\\x0b\\x97\\x1f\\x8f\\xc7o\\x86wZ\\xd4\\x0b+\\xc7L\\x16\\xf6]\\x0f\\ni\\xcc!\\xd2d\\xe1~\\xbb\\xeeN\\xa0X<\\x05\\xbfdB\\x0b\\x1fd\\x18\\xdcvc\\x98E\\xb5/\\xb1\\x85Y\\xa4\\x9c\\xb5\\xe6q\\xc6\\xe4\\xb9\\xcb\\x05\\xa9-\\x8f\\\\\\x93\\x97\\xc6:i.\\x15}\\xe0\\x19Q\\xd2\\xa2\\x0f}\\x7f\\xd4\\xc9G\\x025\\xe6\\x95\\xa4uA\\xad\\x14\\x9f+^\\xfev\\x86\\xa33\\xe2J\\x8a\\xc1hqF\\xff\\xac[\\xd5}\\x01\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xd9T\\x10vI\\x01\\x00\\x00k\\x02\\x00\\x00\\x11\\x00\\x08\\x01docProps/core.xml \\xa2\\x04\\x01(\\xa0\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x8c\\x92_O\\xc3 \\x14\\xc5\\xdfM\\xfc\\x0e\\r\\xef-\\xb4s\\xb3\\x92\\xb6\\x8b\\x7f\\xb2\\'\\x97\\x98X\\xa3\\xd9\\x1b\\x81\\xbb\\xad\\xb1\\x05\\x02h\\xb7o/\\xed\\xb6Z\\x9d\\x0f>\\xc29\\xf7\\xc797d\\xf3]S\\x07\\x9f`l\\xa5d\\x8e\\xe2\\x88\\xa0\\x00$W\\xa2\\x92\\x9b\\x1c\\xbd\\x94\\x8b0E\\x81uL\\nV+\\t9\\xda\\x83E\\xf3\\xe2\\xf2\"\\xe3\\x9are\\xe0\\xc9(\\r\\xc6U`\\x03O\\x92\\x96r\\x9d\\xa3\\xads\\x9abl\\xf9\\x16\\x1af#\\xef\\x90^\\\\+\\xd30\\xe7\\x8ff\\x835\\xe3\\xefl\\x038!d\\x86\\x1bpL0\\xc7p\\x07\\x0c\\xf5@DG\\xa4\\xe0\\x03R\\x7f\\x98\\xba\\x07\\x08\\x8e\\xa1\\x86\\x06\\xa4\\xb38\\x8eb\\xfc\\xedu`\\x1a\\xfb\\xe7@\\xaf\\x8c\\x9cM\\xe5\\xf6\\xdaw:\\xc6\\x1d\\xb3\\x05?\\x88\\x83{g\\xab\\xc1\\xd8\\xb6m\\xd4N\\xfa\\x18>\\x7f\\x8c\\xdf\\x96\\x8f\\xcf}\\xd5\\xb0\\x92\\xdd\\xae8\\xa0\"\\x13\\x9cr\\x03\\xcc)S\\xdc2\\xa3d\\xb0\\x82\\xba\\x96`2<R\\xba-\\xd6\\xcc\\xba\\xa5_\\xf8\\xba\\x02q\\xb7\\xffm>7xr_\\xe4\\x80\\x07\\x11\\xf8h\\xf4P\\xe4\\xa4\\xbcN\\xee\\x1f\\xca\\x05*\\x12\\x92\\x90\\x90\\xdc\\x84\\xf1UIR:\\x9d\\xd1)Yu\\xef\\xff\\x98\\xef\\xa2\\x1e.\\x9ac\\x8a\\xff\\x10\\xaf\\xcb8\\xa6IJI:\"\\x9e\\x00E\\x86\\xcf\\xbeG\\xf1\\x05\\x00\\x00\\xff\\xff\\x03\\x00PK\\x03\\x04\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00aI\\t\\x10\\x89\\x01\\x00\\x00\\x11\\x03\\x00\\x00\\x10\\x00\\x08\\x01docProps/app.xml \\xa2\\x04\\x01(\\xa0\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x9c\\x92Ao\\xdb0\\x0c\\x85\\xef\\x03\\xfa\\x1f\\x0c\\xdd\\x1b9\\xddP\\x0c\\x81\\xacbHW\\xf4\\xb0a\\x01\\x92\\xb6gM\\xa6c\\xa1\\xb2$\\x88\\xac\\x91\\xec\\xd7\\x8f\\xb6\\xd1\\xd4\\xd9z\\xea\\x8d\\xe4{x\\xfaDI\\xdd\\x1c:_\\xf4\\x90\\xd1\\xc5P\\x89\\xe5\\xa2\\x14\\x05\\x04\\x1bk\\x17\\xf6\\x95x\\xd8\\xdd]~\\x15\\x05\\x92\\t\\xb5\\xf11@%\\x8e\\x80\\xe2F_|R\\x9b\\x1c\\x13dr\\x80\\x05G\\x04\\xacDK\\x94VR\\xa2m\\xa13\\xb8`9\\xb0\\xd2\\xc4\\xdc\\x19\\xe26\\xefel\\x1ag\\xe16\\xda\\x97\\x0e\\x02\\xc9\\xab\\xb2\\xbc\\x96p \\x085\\xd4\\x97\\xe9\\x14(\\xa6\\xc4UO\\x1f\\r\\xad\\xa3\\x1d\\xf8\\xf0qwL\\x0c\\xac\\xd5\\xb7\\x94\\xbc\\xb3\\x86\\xf8\\x96\\xfa\\xa7\\xb39bl\\xa8\\xf8~\\xb0\\xe0\\x95\\x9c\\x8b\\x8a\\xe9\\xb6`_\\xb2\\xa3\\xa3.\\x95\\x9c\\xb7jk\\x8d\\x875\\x07\\xeb\\xc6x\\x04%\\xdf\\x06\\xea\\x1e\\xcc\\xb0\\xb4\\x8dq\\x19\\xb5\\xeai\\xd5\\x83\\xa5\\x98\\x0bt\\x7fxmW\\xa2\\xf8m\\x10\\x06\\x9cJ\\xf4&;\\x13\\x88\\xb1\\x06\\xdb\\xd4\\x8c\\xb5OHY?\\xc5\\xfc\\x8c-\\x00\\xa1\\x92l\\x98\\x86c9\\xf7\\xcek\\xf7E/G\\x03\\x17\\xe7\\xc6!`\\x02a\\xe1\\x1cq\\xe7\\xc8\\x03\\xfej6&\\xd3;\\xc4\\xcb9\\xf1\\xc80\\xf1N8\\xdb\\x81o:s\\xce7^\\x99O\\xfa\\'{\\x1d\\xbbd\\xc2\\x91\\x85S\\xf5\\xc3\\x85g|H\\xbbxk\\x08^\\xd7y>T\\xdb\\xd6d\\xa8\\xf9\\x05N\\xeb>\\r\\xd4=o2\\xfb!d\\xdd\\x9a\\xb0\\x87\\xfa\\xd5\\xf3\\xbf0<\\xfe\\xe3\\xf4\\xc3\\xf5\\xf2zQ~.\\xf9]g3%\\xdf\\xfe\\xb2\\xfe\\x0b\\x00\\x00\\xff\\xff\\x03\\x00PK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00b\\xee\\x9dh^\\x01\\x00\\x00\\x90\\x04\\x00\\x00\\x13\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00[Content_Types].xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xb5U0#\\xf4\\x00\\x00\\x00L\\x02\\x00\\x00\\x0b\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x97\\x03\\x00\\x00_rels/.relsPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xbd\\x92e\\xe3W\\x03\\x00\\x00+\\x08\\x00\\x00\\x0f\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xbc\\x06\\x00\\x00xl/workbook.xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\x81>\\x94\\x97\\xf3\\x00\\x00\\x00\\xba\\x02\\x00\\x00\\x1a\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00@\\n\\x00\\x00xl/_rels/workbook.xml.relsPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00WJ\\x84%r\\x06\\x00\\x00\\xfb&\\x00\\x00\\x18\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00s\\x0c\\x00\\x00xl/worksheets/sheet1.xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xc1\\x17\\x10\\xbeN\\x07\\x00\\x00\\xc6 \\x00\\x00\\x13\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x1b\\x13\\x00\\x00xl/theme/theme1.xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xc9=c\\xbf\\xd1\\x02\\x00\\x00\\x0b\\x07\\x00\\x00\\r\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x9a\\x1a\\x00\\x00xl/styles.xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\x1c&\\xa3z\\x9b\\x00\\x00\\x00\\xb4\\x00\\x00\\x00\\x14\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x96\\x1d\\x00\\x00xl/sharedStrings.xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00\\xd9T\\x10vI\\x01\\x00\\x00k\\x02\\x00\\x00\\x11\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00c\\x1e\\x00\\x00docProps/core.xmlPK\\x01\\x02-\\x00\\x14\\x00\\x06\\x00\\x08\\x00\\x00\\x00!\\x00aI\\t\\x10\\x89\\x01\\x00\\x00\\x11\\x03\\x00\\x00\\x10\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\xe3 \\x00\\x00docProps/app.xmlPK\\x05\\x06\\x00\\x00\\x00\\x00\\n\\x00\\n\\x00\\x80\\x02\\x00\\x00\\xa2#\\x00\\x00\\x00\\x00'}"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TcY9zW1tbgf_",
        "colab_type": "text"
      },
      "source": [
        "The decription of the medline format is [here](https://www.nlm.nih.gov/bsd/disted/pubmedtutorial/030_080.html). You can parse this *handle* using the **Medline.parse** function. More informaton is [here](https://biopython.org/DIST/docs/api/Bio.Medline-module.html).\n",
        "\n",
        "Once you grab enough information about each article from pubmed, your task is to train a model uisng the given data and make predictions for the articles in test data (\"diseases-test-without-labels.csv\"). In order to find the best parameter values for your models, you will split the given data into train-1 and train-2. Then you will use train-1 to train your models and train-2 to test your models (typically, train-2 is called the dev set or validation set). This way, you are able to find the best model for making final predictions. \n",
        "\n",
        "Please answer the following questions. **You are required to have at least one code block followed by a single text block for each question.** The code block(s) should contain your code for generating the answer to the corresponding question. The text block should have the actual answer to the question.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "H9No8gvCdXv1",
        "colab_type": "text"
      },
      "source": [
        "1. Details about datasets: What is the label distribution in the full dataset? What are the sizes of train-1 and train-2 datasets you used (and their individual label distriutions)? Also, show the distribution(s) visually.\n",
        "\n",
        "(Note: you can create bar plots or pie charts using the following:\n",
        "[matplotlib.pyplot.bar](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.bar.html), \n",
        "[matplotlib.pyplot.pie](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html)) \n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "b8Wc_PikfgTV",
        "colab_type": "text"
      },
      "source": [
        "Train_1 set has 720 data values and Train_2 set has 180 data values. The distributions of the data can be show below. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eL57sC7mPzJE",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 500
        },
        "outputId": "2eb71ca3-4436-4c91-a579-08d68e04a50c"
      },
      "source": [
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import time\n",
        "from Bio import Medline, Entrez\n",
        "import numpy as np\n",
        "from Bio import Medline, Entrez\n",
        "from nltk.stem import LancasterStemmer\n",
        "from nltk.tokenize import word_tokenize\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "import re\n",
        "import matplotlib as plt\n",
        "Data =  pd.read_csv('diseases-train.csv')\n",
        "train_1, train_2 = train_test_split(Data, test_size=0.2, random_state=42)\n",
        "train_1.category.hist(bins=50, figsize=(10,5))\n",
        "plt.show()"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-9-df21defa3458>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0mtrain_1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_2\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_test_split\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mData\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtest_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m42\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0mtrain_1\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcategory\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbins\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m50\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m: module 'matplotlib' has no attribute 'show'"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gTKfkntdfUgA",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 483
        },
        "outputId": "41858763-1f55-4584-bcb4-3f9f40aa3bae"
      },
      "source": [
        "train_2.category.hist(bins=50, figsize=(10,5))\n",
        "plt.show()"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-11-067182f1a6e0>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mtrain_2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcategory\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbins\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m50\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mfigsize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m: module 'matplotlib' has no attribute 'show'"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tPq7cQJVVtQq",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "e7a2fa97-8163-4254-d4f4-8a262f4f1bdc"
      },
      "source": [
        "len(train_1)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "720"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "I44suZUfD-Fe",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "3bc8a272-4f7e-434a-afee-f8647a5c67eb"
      },
      "source": [
        "len(train_2)"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "180"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EbIZIE3YP1SU",
        "colab_type": "text"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "m0BB-Y5jskIR",
        "colab_type": "text"
      },
      "source": [
        "2. Data preprocessing:\n",
        "How did you preprocess you data? You can do stemming/lemmatization using [NLTK](https://www.nltk.org/) library:\n",
        "[stemming](https://www.nltk.org/api/nltk.stem.html), \n",
        "[lemmatization](https://www.nltk.org/_modules/nltk/stem/wordnet.html). Make sure to apply the same pre-processing to both train-1 and train-2.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yn2XxQM0QqLj",
        "colab_type": "text"
      },
      "source": [
        "For data preprocessing i took a stemming approach using the lancaster stemmer from the nltk library."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r2hpkzqwQpFz",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "Entrez.email = \"aaronzellner1234@gmail.com\"\n",
        "def clean_data(Data_file):\n",
        "  edit_String = \"\"\n",
        "  data_list = []\n",
        "  for index, data in Data_file.iterrows():\n",
        "      doc_1 = data[0]\n",
        "      doc_2 = Entrez.efetch(db=\"pubmed\", id=doc_1, rettype=\"medline\", retmode=\"text\")\n",
        "      journals = Medline.read(doc_2)\n",
        "      new_data = pre_processor(journals)\n",
        "      data_list.append(new_data)\n",
        "  return data_list\n",
        "\n",
        "\n",
        "def pre_processor(info):\n",
        "  Analyze = LancasterStemmer()\n",
        "  edit_String = \"\"\n",
        "  new_string = \"\"\n",
        "  word_list = []\n",
        "  for key, values in info.items():\n",
        "    if key == 'AB':\n",
        "      values = ''.join(str(value) for value in values)\n",
        "      new_values = re.sub(\"[^A-Za-z0-9]\",  \" \", values)\n",
        "      new_words = word_tokenize(new_values)\n",
        "      for word in new_words: \n",
        "            word = Analyze.stem(word)\n",
        "            word_list.append(word)\n",
        "      adjusted = ' '.join(w for w in word_list)\n",
        "      new_string = new_string + adjusted\n",
        "  return new_string \n",
        "\n",
        "\n",
        "Data_ix = clean_data(train_1)\n",
        "Data_x = clean_data(train_2)"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5SU27Xt2vQAH",
        "colab_type": "text"
      },
      "source": [
        "3. Features used:\n",
        "What feature model was used? Chekout all the options for [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). Also, instead of the Bag-of-words (CountVectorizer) model you can optionally use [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). You can read more about both these approches [here](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction). Regarless of the Vectorizer used, list all paramer values you used.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PKJDdcTXgU_W",
        "colab_type": "text"
      },
      "source": [
        "For the featured model i used the CountVectorizer model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aN0ksk2rgr_T",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "count = CountVectorizer(stop_words=\"english\")\n",
        "# making the second training set smaller so that it could fit with the data set from the file.\n",
        "test_1, test_2 = train_test_split(train_2, test_size=0.55, random_state=42)\n",
        "Test_set_2 = clean_data(test_2)\n"
      ],
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0pMqEeDkS21S",
        "colab_type": "text"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XgFJcO6gwVqJ",
        "colab_type": "text"
      },
      "source": [
        "4. NB Model performance:\n",
        "Report the performance values using Naive Bayes here. What is represented by *alpha* and *fit_prior* parameters? What value pair for *alpha* (try 0.0 or 1.0) and *fit_prior* (try True or False) parameters gives you the best overall performance in terms of macro-avreaged F1 (train using train-1 and test using train-2). For the best performing model, show confusion matrix. Show individual F1 values for each category in a bar chart. What are the categories that are easiest/ hardest to predict?\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xcX_chQ4veQp",
        "colab_type": "text"
      },
      "source": [
        "The performance values of the Naive Bayes was alpha = 1.0 and fit_prior was set equal to true. I answered the rest of this question in question 5\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xQYzI4fkqNbr",
        "colab_type": "code",
        "colab": {
          "resources": {
            "http://localhost:8080/nbextensions/google.colab/files.js": {
              "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7CgpmdW5jdGlvbiBfdXBsb2FkRmlsZXMoaW5wdXRJZCwgb3V0cHV0SWQpIHsKICBjb25zdCBzdGVwcyA9IHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCk7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICAvLyBDYWNoZSBzdGVwcyBvbiB0aGUgb3V0cHV0RWxlbWVudCB0byBtYWtlIGl0IGF2YWlsYWJsZSBmb3IgdGhlIG5leHQgY2FsbAogIC8vIHRvIHVwbG9hZEZpbGVzQ29udGludWUgZnJvbSBQeXRob24uCiAgb3V0cHV0RWxlbWVudC5zdGVwcyA9IHN0ZXBzOwoKICByZXR1cm4gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpOwp9CgovLyBUaGlzIGlzIHJvdWdobHkgYW4gYXN5bmMgZ2VuZXJhdG9yIChub3Qgc3VwcG9ydGVkIGluIHRoZSBicm93c2VyIHlldCksCi8vIHdoZXJlIHRoZXJlIGFyZSBtdWx0aXBsZSBhc3luY2hyb25vdXMgc3RlcHMgYW5kIHRoZSBQeXRob24gc2lkZSBpcyBnb2luZwovLyB0byBwb2xsIGZvciBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcC4KLy8gVGhpcyB1c2VzIGEgUHJvbWlzZSB0byBibG9jayB0aGUgcHl0aG9uIHNpZGUgb24gY29tcGxldGlvbiBvZiBlYWNoIHN0ZXAsCi8vIHRoZW4gcGFzc2VzIHRoZSByZXN1bHQgb2YgdGhlIHByZXZpb3VzIHN0ZXAgYXMgdGhlIGlucHV0IHRvIHRoZSBuZXh0IHN0ZXAuCmZ1bmN0aW9uIF91cGxvYWRGaWxlc0NvbnRpbnVlKG91dHB1dElkKSB7CiAgY29uc3Qgb3V0cHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKG91dHB1dElkKTsKICBjb25zdCBzdGVwcyA9IG91dHB1dEVsZW1lbnQuc3RlcHM7CgogIGNvbnN0IG5leHQgPSBzdGVwcy5uZXh0KG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSk7CiAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShuZXh0LnZhbHVlLnByb21pc2UpLnRoZW4oKHZhbHVlKSA9PiB7CiAgICAvLyBDYWNoZSB0aGUgbGFzdCBwcm9taXNlIHZhbHVlIHRvIG1ha2UgaXQgYXZhaWxhYmxlIHRvIHRoZSBuZXh0CiAgICAvLyBzdGVwIG9mIHRoZSBnZW5lcmF0b3IuCiAgICBvdXRwdXRFbGVtZW50Lmxhc3RQcm9taXNlVmFsdWUgPSB2YWx1ZTsKICAgIHJldHVybiBuZXh0LnZhbHVlLnJlc3BvbnNlOwogIH0pOwp9CgovKioKICogR2VuZXJhdG9yIGZ1bmN0aW9uIHdoaWNoIGlzIGNhbGxlZCBiZXR3ZWVuIGVhY2ggYXN5bmMgc3RlcCBvZiB0aGUgdXBsb2FkCiAqIHByb2Nlc3MuCiAqIEBwYXJhbSB7c3RyaW5nfSBpbnB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIGlucHV0IGZpbGUgcGlja2VyIGVsZW1lbnQuCiAqIEBwYXJhbSB7c3RyaW5nfSBvdXRwdXRJZCBFbGVtZW50IElEIG9mIHRoZSBvdXRwdXQgZGlzcGxheS4KICogQHJldHVybiB7IUl0ZXJhYmxlPCFPYmplY3Q+fSBJdGVyYWJsZSBvZiBuZXh0IHN0ZXBzLgogKi8KZnVuY3Rpb24qIHVwbG9hZEZpbGVzU3RlcChpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IGlucHV0RWxlbWVudCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKGlucHV0SWQpOwogIGlucHV0RWxlbWVudC5kaXNhYmxlZCA9IGZhbHNlOwoKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIG91dHB1dEVsZW1lbnQuaW5uZXJIVE1MID0gJyc7CgogIGNvbnN0IHBpY2tlZFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgaW5wdXRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NoYW5nZScsIChlKSA9PiB7CiAgICAgIHJlc29sdmUoZS50YXJnZXQuZmlsZXMpOwogICAgfSk7CiAgfSk7CgogIGNvbnN0IGNhbmNlbCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2J1dHRvbicpOwogIGlucHV0RWxlbWVudC5wYXJlbnRFbGVtZW50LmFwcGVuZENoaWxkKGNhbmNlbCk7CiAgY2FuY2VsLnRleHRDb250ZW50ID0gJ0NhbmNlbCB1cGxvYWQnOwogIGNvbnN0IGNhbmNlbFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgY2FuY2VsLm9uY2xpY2sgPSAoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9OwogIH0pOwoKICAvLyBXYWl0IGZvciB0aGUgdXNlciB0byBwaWNrIHRoZSBmaWxlcy4KICBjb25zdCBmaWxlcyA9IHlpZWxkIHsKICAgIHByb21pc2U6IFByb21pc2UucmFjZShbcGlja2VkUHJvbWlzZSwgY2FuY2VsUHJvbWlzZV0pLAogICAgcmVzcG9uc2U6IHsKICAgICAgYWN0aW9uOiAnc3RhcnRpbmcnLAogICAgfQogIH07CgogIGNhbmNlbC5yZW1vdmUoKTsKCiAgLy8gRGlzYWJsZSB0aGUgaW5wdXQgZWxlbWVudCBzaW5jZSBmdXJ0aGVyIHBpY2tzIGFyZSBub3QgYWxsb3dlZC4KICBpbnB1dEVsZW1lbnQuZGlzYWJsZWQgPSB0cnVlOwoKICBpZiAoIWZpbGVzKSB7CiAgICByZXR1cm4gewogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgICAgfQogICAgfTsKICB9CgogIGZvciAoY29uc3QgZmlsZSBvZiBmaWxlcykgewogICAgY29uc3QgbGkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdsaScpOwogICAgbGkuYXBwZW5kKHNwYW4oZmlsZS5uYW1lLCB7Zm9udFdlaWdodDogJ2JvbGQnfSkpOwogICAgbGkuYXBwZW5kKHNwYW4oCiAgICAgICAgYCgke2ZpbGUudHlwZSB8fCAnbi9hJ30pIC0gJHtmaWxlLnNpemV9IGJ5dGVzLCBgICsKICAgICAgICBgbGFzdCBtb2RpZmllZDogJHsKICAgICAgICAgICAgZmlsZS5sYXN0TW9kaWZpZWREYXRlID8gZmlsZS5sYXN0TW9kaWZpZWREYXRlLnRvTG9jYWxlRGF0ZVN0cmluZygpIDoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ24vYSd9IC0gYCkpOwogICAgY29uc3QgcGVyY2VudCA9IHNwYW4oJzAlIGRvbmUnKTsKICAgIGxpLmFwcGVuZENoaWxkKHBlcmNlbnQpOwoKICAgIG91dHB1dEVsZW1lbnQuYXBwZW5kQ2hpbGQobGkpOwoKICAgIGNvbnN0IGZpbGVEYXRhUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICAgIGNvbnN0IHJlYWRlciA9IG5ldyBGaWxlUmVhZGVyKCk7CiAgICAgIHJlYWRlci5vbmxvYWQgPSAoZSkgPT4gewogICAgICAgIHJlc29sdmUoZS50YXJnZXQucmVzdWx0KTsKICAgICAgfTsKICAgICAgcmVhZGVyLnJlYWRBc0FycmF5QnVmZmVyKGZpbGUpOwogICAgfSk7CiAgICAvLyBXYWl0IGZvciB0aGUgZGF0YSB0byBiZSByZWFkeS4KICAgIGxldCBmaWxlRGF0YSA9IHlpZWxkIHsKICAgICAgcHJvbWlzZTogZmlsZURhdGFQcm9taXNlLAogICAgICByZXNwb25zZTogewogICAgICAgIGFjdGlvbjogJ2NvbnRpbnVlJywKICAgICAgfQogICAgfTsKCiAgICAvLyBVc2UgYSBjaHVua2VkIHNlbmRpbmcgdG8gYXZvaWQgbWVzc2FnZSBzaXplIGxpbWl0cy4gU2VlIGIvNjIxMTU2NjAuCiAgICBsZXQgcG9zaXRpb24gPSAwOwogICAgd2hpbGUgKHBvc2l0aW9uIDwgZmlsZURhdGEuYnl0ZUxlbmd0aCkgewogICAgICBjb25zdCBsZW5ndGggPSBNYXRoLm1pbihmaWxlRGF0YS5ieXRlTGVuZ3RoIC0gcG9zaXRpb24sIE1BWF9QQVlMT0FEX1NJWkUpOwogICAgICBjb25zdCBjaHVuayA9IG5ldyBVaW50OEFycmF5KGZpbGVEYXRhLCBwb3NpdGlvbiwgbGVuZ3RoKTsKICAgICAgcG9zaXRpb24gKz0gbGVuZ3RoOwoKICAgICAgY29uc3QgYmFzZTY0ID0gYnRvYShTdHJpbmcuZnJvbUNoYXJDb2RlLmFwcGx5KG51bGwsIGNodW5rKSk7CiAgICAgIHlpZWxkIHsKICAgICAgICByZXNwb25zZTogewogICAgICAgICAgYWN0aW9uOiAnYXBwZW5kJywKICAgICAgICAgIGZpbGU6IGZpbGUubmFtZSwKICAgICAgICAgIGRhdGE6IGJhc2U2NCwKICAgICAgICB9LAogICAgICB9OwogICAgICBwZXJjZW50LnRleHRDb250ZW50ID0KICAgICAgICAgIGAke01hdGgucm91bmQoKHBvc2l0aW9uIC8gZmlsZURhdGEuYnl0ZUxlbmd0aCkgKiAxMDApfSUgZG9uZWA7CiAgICB9CiAgfQoKICAvLyBBbGwgZG9uZS4KICB5aWVsZCB7CiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICB9CiAgfTsKfQoKc2NvcGUuZ29vZ2xlID0gc2NvcGUuZ29vZ2xlIHx8IHt9OwpzY29wZS5nb29nbGUuY29sYWIgPSBzY29wZS5nb29nbGUuY29sYWIgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYi5fZmlsZXMgPSB7CiAgX3VwbG9hZEZpbGVzLAogIF91cGxvYWRGaWxlc0NvbnRpbnVlLAp9Owp9KShzZWxmKTsK",
              "ok": true,
              "headers": [
                [
                  "content-type",
                  "application/javascript"
                ]
              ],
              "status": 200,
              "status_text": ""
            }
          },
          "base_uri": "https://localhost:8080/",
          "height": 109
        },
        "outputId": "65ec9300-57b7-466f-cfab-5cbc7750ce25"
      },
      "source": [
        "from google.colab import files\n",
        "files.upload()"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-d0c62eac-b79c-4ac8-a3cd-c319f572600a\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-d0c62eac-b79c-4ac8-a3cd-c319f572600a\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script src=\"/nbextensions/google.colab/files.js\"></script> "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Saving Test_set.csv to Test_set.csv\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'Test_set.csv': b'pmid,category\\r\\n30173950,Eye Diseases\\r\\n30179123,Eye Diseases\\r\\n30147943,Skin Diseases\\r\\n30157597,Skin Diseases\\r\\n30127183,Skin Diseases\\r\\n30145376,Skin Diseases\\r\\n30189294,Skin Diseases\\r\\n30195791,Eye Diseases\\r\\n30168550,Brain Diseases\\r\\n30144345,Skin Diseases\\r\\n30177891,Heart Diseases\\r\\n30174588,Brain Diseases\\r\\n30173580,Brain Diseases\\r\\n30142763,Heart Diseases\\r\\n30189044,Skin Diseases\\r\\n30148164,Eye Diseases\\r\\n30178578,Heart Diseases\\r\\n30127455,Skin Diseases    \\r\\n30125148,Skin Diseases\\r\\n30194482,Eye Diseases\\r\\n30145692,Heart Diseases\\r\\n30144530,Brain Diseases\\r\\n30168088,Eye Diseases\\r\\n30176756,Brain Diseases\\r\\n30179320,Brain Diseases\\r\\n30165439,Skin Diseases\\r\\n30187490,Skin Diseases\\r\\n30120706,Heart Diseases\\r\\n30176735,Skin Diseases\\r\\n30127151,Skin Diseases\\r\\n30166295,Heart Diseases\\r\\n30128158,Eye Diseases\\r\\n30112674,Skin Diseases\\r\\n30121124,Heart Diseases\\r\\n30177139,Heart Diseases\\r\\n30149799,Skin Diseases\\r\\n30136732,Brain Diseases\\r\\n30170414,Skin Diseases\\r\\n30176661,Heart Diseases\\r\\n30182844,Eye Diseases\\r\\n30136733,Skin Diseases\\r\\n30180191,Eye Diseases\\r\\n30138455,Heart Diseases\\r\\n30159858,Heart Diseases\\r\\n30133701,Skin Diseases\\r\\n30127167,Eye Diseases\\r\\n30179143,Skin Diseases\\r\\n30158657,Skin Diseases\\r\\n30107838,Skin Diseases\\r\\n30149095,Skin Diseases\\r\\n30155789,Skin Diseases\\r\\n30169483,Heart Diseases\\r\\n30111354,Skin Diseases\\r\\n30181596,Heart Diseases\\r\\n30196252,Eye Diseases\\r\\n30152403,Eye Diseases\\r\\n30182799,Skin Diseases\\r\\n30123847,Eye Diseases\\r\\n30156120,Eye Diseases\\r\\n30151368,Eye Diseases\\r\\n30166200,Brain Diseases\\r\\n30123404,Brain Diseases\\r\\n30157483,Brain Diseases\\r\\n30176352,Eye Diseases\\r\\n30174885,Brain Diseases\\r\\n30184550,Heart Diseases\\r\\n30095615,Brain Diseases\\r\\n30165597,Brain Diseases\\r\\n30127536,Eye Diseases\\r\\n30187287,Heart Diseases\\r\\n30170463,Skin Diseases\\r\\n30167679,Skin Diseases\\r\\n30135134,Brain Diseases\\r\\n30191431,Heart Diseases\\r\\n30151401,Heart Diseases\\r\\n30187665,Skin Diseases\\r\\n30094716,Brain Diseases\\r\\n30142836,Heart Diseases\\r\\n30107649,Brain Diseases\\r\\n30177082,Skin Diseases\\r\\n30175978,Brain Diseases\\r\\n30181795,Eye Diseases\\r\\n30153390,Skin Diseases\\r\\n30165544,Skin Diseases\\r\\n30158120,Skin Diseases\\r\\n30173663,Skin Diseases\\r\\n30187290,Skin Diseases\\r\\n30128298,Brain Diseases\\r\\n30121712,Heart Diseases\\r\\n30142002,Skin Diseases\\r\\n30170545,Eye Diseases\\r\\n30174119,Brain Diseases \\r\\n30177223,Heart Diseases\\r\\n30169897,Brain Diseases\\r\\n30151293,Brain Diseases\\r\\n30172283,Skin Diseases\\r\\n30144097,Heart Diseases\\r\\n30178211,Skin Diseases\\r\\n30134277,Brain Diseases\\r\\n30158522,Eye Diseases\\r\\n'}"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LEN7-fYGhS-j",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "a992adac-cb4e-4541-aee9-96534ee7873c"
      },
      "source": [
        "#fitting the model with the first trianing set and making a prediction\n",
        "evaluate = MultinomialNB()\n",
        "fit_train_1 = count.fit_transform(Data_ix)\n",
        "evaluate.fit(fit_train_1, train_1.category)\n",
        "evaluate.predict(fit_train_1)\n",
        "\n",
        "#Test File Data\n",
        "new_file = pd.read_csv(\"Test_set.csv\")\n",
        "Data_test = clean_data(new_file)\n",
        "test_fit = count.fit_transform(Data_test)\n",
        "evaluate.fit(test_fit, test_2.category)"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "colab_type": "text",
        "id": "tGC-5yYqVTX2"
      },
      "source": [
        "5. Other Model performance:\n",
        "Comapre NB performance to at least one other model mentioned in the class (e.g. KNN). Which model did you pick? List all parameter values you selected. Show the individual F1 values for each category for the two models side-by-side in a bar chart.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "58mu1BCAfaZ_",
        "colab_type": "text"
      },
      "source": [
        "The F1 score is: 0.9445225323228315"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "643jSSgrfc4F",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        },
        "outputId": "fc559744-52e3-4c14-c8f7-9158844808f8"
      },
      "source": [
        "from sklearn import metrics\n",
        "prediction = evaluate.predict(test_fit)\n",
        "print(\"F_1 Score for Model NB is: \", metrics.f1_score(test_2.category, prediction, average=\"macro\"))\n",
        "from sklearn.svm import SVC\n",
        "svm = SVC()\n",
        "svm.fit(test_fit, test_2.category)\n",
        "svm.predict(test_fit)\n",
        "second_predictions = svm.predict(test_fit)\n",
        "print(\"F_1 Score for SVM model is: \", metrics.f1_score(test_2.category, second_predictions, average=\"macro\"))"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "F_1 Score for Model NB is:  0.9445225323228315\n",
            "F_1 Score for SVM model is:  0.9290584415584415\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ogqd5yG4rrnA",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        },
        "outputId": "249a53c4-f838-47de-abf5-0d19e8e1ea6e"
      },
      "source": [
        "print(metrics.classification_report(test_2.category, prediction), metrics.confusion_matrix(test_2.category, prediction))"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "                precision    recall  f1-score   support\n",
            "\n",
            "Brain Diseases       1.00      0.88      0.94        25\n",
            "  Eye Diseases       1.00      0.95      0.98        21\n",
            "Heart Diseases       1.00      0.91      0.95        23\n",
            " Skin Diseases       0.84      1.00      0.91        31\n",
            "\n",
            "      accuracy                           0.94       100\n",
            "     macro avg       0.96      0.94      0.94       100\n",
            "  weighted avg       0.95      0.94      0.94       100\n",
            " [[22  0  0  3]\n",
            " [ 0 20  0  1]\n",
            " [ 0  0 21  2]\n",
            " [ 0  0  0 31]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Telu6cOwsxqD",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        },
        "outputId": "7c3c8cba-9bb5-4463-fa9e-37baa38eb8ea"
      },
      "source": [
        "print(metrics.classification_report(test_2.category, second_predictions), metrics.confusion_matrix(test_2.category, second_predictions))"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "                precision    recall  f1-score   support\n",
            "\n",
            "Brain Diseases       0.83      1.00      0.91        25\n",
            "  Eye Diseases       1.00      0.90      0.95        21\n",
            "Heart Diseases       1.00      0.83      0.90        23\n",
            " Skin Diseases       0.94      0.97      0.95        31\n",
            "\n",
            "      accuracy                           0.93       100\n",
            "     macro avg       0.94      0.92      0.93       100\n",
            "  weighted avg       0.94      0.93      0.93       100\n",
            " [[25  0  0  0]\n",
            " [ 2 19  0  0]\n",
            " [ 2  0 19  2]\n",
            " [ 1  0  0 30]]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "82N-WtSqtbl4",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "outputId": "3dd97678-b151-4112-ea9d-87ad5b8b4d1b"
      },
      "source": [
        "\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "plt.title(\"SVC Model\")\n",
        "categories = [\"Brain Diseases\", \"Eye Diseases\", \"Heart Diseases\", \"Skin Diseases\"]\n",
        "F_1Scores = [0.91, 0.95, 0.90, 0.95]\n",
        "plt.bar(categories, F_1Scores)"
      ],
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<BarContainer object of 4 artists>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 37
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAEICAYAAABPgw/pAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAWCklEQVR4nO3dedRlVX3m8e9DlTgxmFilSym0UDFKMBGtRo0TKhrQFpLWKBiCJEZi1kLi0HZwSViEtjsqcVhRkiVEBRFFHFORUnAijiCFyFhCqhlCId0WikRQmfz1H2e/cLm8Ve99q25RVZvvZ6131Tn77HvOPvve+9x99h0qVYUkaeu3zeZugCRpOgx0SeqEgS5JnTDQJakTBrokdcJAl6ROGOjSJpRkaZJKsnCCuock+da90S71yUDXFivJs5J8J8mNSX6a5NtJ/kuSpye5Ocl2s9zm/CSHteVtkxyd5N9b/auSfDjJ0nUc76oktyZZNMs+a123k7YUBrq2SEl2AL4AvB/4TWAn4G+BW6rqbGAN8PKx2+wO7AZ8ohV9GtgPeBWwI/C7wHnAC9Zz6CuBA0f2+STgQRt/RtKmZ6BrS/V4gKr6RFXdUVW/rKozq+rCtv0k4OCx2xwMrKiqnyTZG3ghsH9VnVtVt1fVjVV1XFV9aD3HPXlsv68GPjpaIcmOST6aZG2Sq5McmWSbtm1Bkr9Pcn2SK4CXzHLbDyW5Lsm1Sd6eZME8+0aalYGuLdXlwB1JTkqyb5LfGNt+MvCcJDsDtEB9FUPQA+wNfK+qrpnncc8GdkjyxBa0BwAfG6vzfoYR/2OA5zK8APxp2/Za4L8CewDLGLuKAE4Ebgce1+q8CPjzebZRmpWBri1SVf0n8CyggBOAtUmWJ3l4234NcBbwJ+0mLwDuD5ze1h8KXLeBh58Zpb8QWAVcO7NhJOTfWlU/r6qrgHePtOMVwPuq6pqq+inwdyO3fTjwYuANVXVzVf0YeG/bn7TRDHRtsapqVVUdUlVLgN2BRwLvG6lyEncF6Z8Ap1bVbW39J8AjNvDQJzOM9g9hbLoFWATcD7h6pOxqhjl+WhuvGds249Htttcl+VmSnwEfBB62ge2U7sZA11ahqn7IMF2x+0jxZ4ElSZ4H/Dfumm4B+AqwZ5IlG3CsqxneHH1xO8ao64HbGMJ5xqO4axR/HbDz2LYZ1wC3AIuq6iHtb4eq+u35tlGajYGuLVKSJyR580wgt7nyAxnmuAGoqpsZPsnyEeDqqlo5su0rwJeBzyV5apKFSbZP8rokfzZBE14DPL8d405VdQdwGvC/2v4eDbyJu+bZTwMOT7KkzfsfMXLb64AzgXcn2SHJNkkem+S58+sdaXYGurZUPweeBpyT5GaGIL8YePNYvZMYRsvjUyMwvCG5AvgkcGO7/TKG0ft6VdX/GX2BGPN64GbgCuBbwMeBD7dtJwBnABcA3+eeI/yDgW2BS4EbGF6QNnRqSLqb+B9cSFIfHKFLUicMdEnqhIEuSZ0w0CWpE3P+pOemsmjRolq6dOnmOrwkbZXOO++866tq8WzbNlugL126lJUr1/WpMEnSbJJcva5tTrlIUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InNts3RbX5LD3i9Lkrdeyqd7xkczfhPs/H4KZ5DDpCl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ0w0CWpE1vl59D9DKufo5Z0T47QJakTBrokdWKrnHKRNqf7+pQfOO23pXKELkmdMNAlqRMGuiR1wkCXpE4Y6JLUCQNdkjphoEtSJwx0SeqEgS5JnTDQJakTBrokdcJAl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ2YKNCT7JPksiSrkxwxy/ZHJfl6kvOTXJjkxdNvqiRpfeYM9CQLgOOAfYHdgAOT7DZW7UjgtKraAzgA+MdpN1SStH6TjND3BFZX1RVVdStwKrD/WJ0CdmjLOwI/ml4TJUmTmCTQdwKuGVlf08pGHQ0clGQNsAJ4/Ww7SnJokpVJVq5du3YDmitJWpdpvSl6IHBiVS0BXgycnOQe+66q46tqWVUtW7x48ZQOLUmCyQL9WmDnkfUlrWzUa4DTAKrqu8ADgEXTaKAkaTKTBPq5wK5JdkmyLcObnsvH6vwH8AKAJE9kCHTnVCTpXjRnoFfV7cBhwBnAKoZPs1yS5Jgk+7VqbwZem+QC4BPAIVVVm6rRkqR7WjhJpapawfBm52jZUSPLlwLPnG7TJEnz4TdFJakTBrokdcJAl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InDHRJ6oSBLkmdMNAlqRMGuiR1wkCXpE4Y6JLUCQNdkjphoEtSJwx0SeqEgS5JnTDQJakTBrokdcJAl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekTkwU6En2SXJZktVJjlhHnVckuTTJJUk+Pt1mSpLmsnCuCkkWAMcBLwTWAOcmWV5Vl47U2RV4K/DMqrohycM2VYMlSbObZIS+J7C6qq6oqluBU4H9x+q8Fjiuqm4AqKofT7eZkqS5TBLoOwHXjKyvaWWjHg88Psm3k5ydZJ/ZdpTk0CQrk6xcu3bthrVYkjSrab0puhDYFdgLOBA4IclDxitV1fFVtayqli1evHhKh5YkwWSBfi2w88j6klY2ag2wvKpuq6orgcsZAl6SdC+ZJNDPBXZNskuSbYEDgOVjdT7PMDonySKGKZgrpthOSdIc5gz0qrodOAw4A1gFnFZVlyQ5Jsl+rdoZwE+SXAp8HXhLVf1kUzVaknRPc35sEaCqVgArxsqOGlku4E3tT5K0GfhNUUnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InDHRJ6oSBLkmdMNAlqRMGuiR1wkCXpE4Y6JLUCQNdkjphoEtSJwx0SeqEgS5JnTDQJakTBrokdcJAl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InDHRJ6oSBLkmdmCjQk+yT5LIkq5McsZ56L0tSSZZNr4mSpEnMGehJFgDHAfsCuwEHJtltlnrbA38FnDPtRkqS5jbJCH1PYHVVXVFVtwKnAvvPUu9/Au8EfjXF9kmSJjRJoO8EXDOyvqaV3SnJU4Cdq+r0KbZNkjQPG/2maJJtgPcAb56g7qFJViZZuXbt2o09tCRpxCSBfi2w88j6klY2Y3tgd+CsJFcBTweWz/bGaFUdX1XLqmrZ4sWLN7zVkqR7mCTQzwV2TbJLkm2BA4DlMxur6saqWlRVS6tqKXA2sF9VrdwkLZYkzWrOQK+q24HDgDOAVcBpVXVJkmOS7LepGyhJmszCSSpV1QpgxVjZUeuou9fGN0uSNF9+U1SSOmGgS1InDHRJ6oSBLkmdMNAlqRMGuiR1wkCXpE4Y6JLUCQNdkjphoEtSJwx0SeqEgS5JnTDQJakTBrokdcJAl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InDHRJ6oSBLkmdMNAlqRMGuiR1wkCXpE4Y6JLUCQNdkjphoEtSJwx0SerERIGeZJ8klyVZneSIWba/KcmlSS5M8tUkj55+UyVJ6zNnoCdZABwH7AvsBhyYZLexaucDy6rqd4BPA++adkMlSes3yQh9T2B1VV1RVbcCpwL7j1aoqq9X1S/a6tnAkuk2U5I0l0kCfSfgmpH1Na1sXV4DfHG2DUkOTbIyycq1a9dO3kpJ0pym+qZokoOAZcCxs22vquOrallVLVu8ePE0Dy1J93kLJ6hzLbDzyPqSVnY3SfYG3gY8t6pumU7zJEmTmmSEfi6wa5JdkmwLHAAsH62QZA/gg8B+VfXj6TdTkjSXOQO9qm4HDgPOAFYBp1XVJUmOSbJfq3YssB3wqSQ/SLJ8HbuTJG0ik0y5UFUrgBVjZUeNLO895XZJkubJb4pKUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InDHRJ6oSBLkmdMNAlqRMGuiR1wkCXpE4Y6JLUCQNdkjphoEtSJwx0SeqEgS5JnTDQJakTBrokdcJAl6ROGOiS1AkDXZI6YaBLUicMdEnqhIEuSZ0w0CWpEwa6JHXCQJekThjoktQJA12SOmGgS1InDHRJ6oSBLkmdmCjQk+yT5LIkq5McMcv2+yf5ZNt+TpKl026oJGn95gz0JAuA44B9gd2AA5PsNlbtNcANVfU44L3AO6fdUEnS+k0yQt8TWF1VV1TVrcCpwP5jdfYHTmrLnwZekCTTa6YkaS4LJ6izE3DNyPoa4GnrqlNVtye5EXgocP1opSSHAoe21ZuSXLYhjd4CLGLs3O5N2fqvf+y/jWcfbpytuf8eva4NkwT61FTV8cDx9+YxN4UkK6tq2eZux9bK/tt49uHG6bX/JplyuRbYeWR9SSubtU6ShcCOwE+m0UBJ0mQmCfRzgV2T7JJkW+AAYPlYneXAq9vyy4GvVVVNr5mSpLnMOeXS5sQPA84AFgAfrqpLkhwDrKyq5cCHgJOTrAZ+yhD6Pdvqp402M/tv49mHG6fL/osDaUnqg98UlaROGOiS1ImtItCT3JHkB0kuSPL9JL+3AftYkeQh86h/dJJr23H/PclnR78hm+SfZ/nG7FZhpD9n/u7xcw4bsM8Tk1zZ7qPLk3w0yZKR7fPq/y1JkpvG1g9J8oEp7XtpkletZ9svk5yfZFWS7yU5ZGT7ftO47+5NSd6W5JIkF7bH3tNa+VVJFs1S/zvz3H+3j8OJVNUW/wfcNLL8+8C/zVJn4ZSPeTTw30fWXwn8X2Dx5u6PafbnFPd5IvDythzgjcDlwLab+3yn3V/AIcAHprDfhcBewBfWsX0pcPHI+mOAHwB/urn7ZAPP9xnAd4H7t/VFwCPb8lXAoikco9vH4SR/W8UIfcwOwA0ASfZK8s0ky4FLW9nnk5zXRgEz30q9cwTQRj2rkpzQ6pyZ5IFzHbSqPgmcCbyq7e+sJMuSLGijgouTXJTkjW37Y5N8qbXlm0me0Mpf2n7A7PwkX0ny8Fb+3JER8/lJtm/lb0lybhvR/G0re3CS09so5OIkr9zYTk3y/CSfH1l/YZLPteUXJfluuzr6VJLt5uirqqr3MrwA7tv2MdP/s7Y9yVOT/FvrrzOSPKKVv7ad/wVJPpPkQa38j9rtL0jyjVa2IMmxI/31F638EUm+0fr24iTP3tj+Gumnxa1d57a/Z7byPVufnZ/kO0l+q5UfkmR5kq8BXwXeATy7te2Nc/TrFcCbgMNH9vWBDeiP7ZJ8td2fFyXZv5XP9745PMmlbd+nTtBdjwCur6pb2vlcX1U/GuvPByb5YpLXtvWb2r97tefcp5P8MMkpyfp/XuS+9DgcPekt/g+4g2Fk8kPgRuCprXwv4GZgl5G6v9n+fSBwMfDQGhkBMIx6bgee3MpPAw6a5ZhHMzJCb2VvAP6pLZ8FLAOeCnx5pM5D2r9fBXZty09j+Gw+wG9w16eL/hx4d1v+V+CZbXk7htHbixg+XhWG6bEvAM8BXgacMHLMHTewP2f+XtmO8UPaFQjwceClrc++ATy4lf81cNQs+zyRNjIaKXsf8Ndj/X+PtgP3A74zcuxXMnw8lpn7ry2/HXh9W74I2Gmszw8FjmzL9wdWArsAbwbe1soXANtvZH/9B22E3vrpWW35UcCqtrwD7aoR2Bv4TFs+hOHnM2Yep3sx4Qh95lyBX47s6wMb0B8LgR1a+SJgdbv/53vf/Ii7RtsPmaAft2v9dznwj8BzR7Zd1c73K8DBI+U3jfTTjQxfbNyGYaT/rPvS43CSv3v1q/8b4ZdV9WSAJM8APppk97bte1V15Ujdw5P8YVveGdiVe35r9cqq+kFbPo/hgTSJ2UYEVwCPSfJ+4HTgzDaC/T3gUyODiPu3f5cAn2yv/NsCM23/NvCeJKcAn62qNUlexBDq57c627Xz+Sbw7iTvZAiDb07Y/hl39ufdTi45GTgoyUcYLo8PBvZh+JXNb7dz2ZbhyTSJ2frrovG2t/tyd+DL7RgLgOta/d2TvJ0hyLZj+D4EDP11YpLTgM+2shcBv5Pk5W19R4b+Ohf4cJL7AZ8fue8ndbf+yjCPPfO18b2B3Ubu5x3a/b8jcFKSXYFiCIsZX66qn86zDXcefh3l8+mPNcD/TvIc4NcMv8X0cOZ/31wInJLhyu7Oq7t1qaqbkjwVeDbwPIbnwRFVdWKr8i/Au6rqlHXs4ntVtQYgyQ8Ynrffmuu49PM4nNPWEuh3qqrvZnjzZHErunlmW5K9GJ5gz6iqXyQ5C3jALLu5ZWT5DobR/CT2YHi1HW3PDUl+l2Fu/3XAKxhG8j+bLTSB9wPvqarlrb1Ht/28I8npwIsZwvP3GR6If1dVHxzfSZKntLpvT/LVqjpmwnNYn48wXCn8CvhUDV8qC0MAHbgB+9uD4UrlTlV1+Xjbgc8Bl1TVM2bZx4nAH1TVBS1I92r7eV2GN9ReApzXgiIMI6czxnfSwuslDE++91TVRzfgfGazDfD0qvrV2PE+AHy9qv4ww/8PcNbI5pvZcHsAq8YL59MfrR8XM1zp3pbkKuABG3DfvIThivGlwNuSPKmqbl9f46vqDoa+OCvJRQzfMD+xbf42sE+Sj1cbxo4Zf95Oml/3hcchsJV8ymVUhrnoBcz+WzE7Mvwu+y9avadP8bgvY3jl/cRY+SJgm6r6DHAk8JSq+k/gyiR/1Oqkhf5MG2d+C+fVI/t5bFVdVFXvZHglfwLDKODP2oiPJDsleViSRwK/qKqPAccCT5nGOdYwn/mjdh4facVnA89M8rjWhgcnefz69tPO93CGOdMvjW2bre2XAYvb1RdJ7pfkt9tNtgeua6OaPx7Zz2Or6pyqOgpYy3A1dgbwl60uSR7f2vto4P9V1QnAPzOl/mrOBF4/0q6ZF/HR+/mQ9dz+5wznOKf2wvD3DIOC8W0T90dr249bmD+P9ut987lvkmwD7FxVX2eYhtuRYeS6vvb/VrtimfFk4OqR9aMY3h87bpL+mMt97HEIbD0j9Ae2SywYXv1eXVV35J7viXwJeF2SVQx3ztkbedw3JjkIeDDDfPzzq2rtWJ2dgI+0BzjAW9u/fwz8U5IjGS63TwUuYBiRfyrJDcDXGObWAN7Qnly/Bi4BvlhVtyR5IvDddq43AQcBjwOOTfJr4DbgL+d5XqP9CfClqpr5+NspDHOIqwCqam0bkXwiycy00ZEM86Djjk3yN8CDGPr+eTX8hv6oJ423vapubZen/5BkR4bH5ftaP/wNcA7Dk+Uc7gq/Y1s4hGH0dQHDFMBS4PvtymIt8AcMo6m3JLmNoQ8Pnrin5nY4cFySC1u7v8FwpfYuhimXIxmm4tblQuCOJBcAJ9bwJt6oxyY5n+FK8+fAP4xMUYyaT3+cAvxrGyGvZHjvBOZ331wOfKyVpbXrZ3P01XbA+zN8bPB2hrn7Q8fq/BXDtMS7qup/zLG/dbkvPg4Bv/qvMW2q4Pyq+tDmbouk+THQdack5zHM776w2kfLJG09DHRJ6sRW96aoJGl2BrokdcJAl6ROGOiS1AkDXZI68f8BA01/4+UGTocAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jq7zdzo-u41a",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "outputId": "24c6ad5f-6dfc-4ea7-e1f4-dad22b16f077"
      },
      "source": [
        "plt.title(\"Multinomial NB Model\")\n",
        "categories = [\"Brain Diseases\", \"Eye Diseases\", \"Heart Diseases\", \"Skin Diseases\"]\n",
        "F_1Scores = [0.94, 0.98, 0.95, 0.91]\n",
        "plt.bar(categories, F_1Scores)"
      ],
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<BarContainer object of 4 artists>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 36
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "psjuM8gvejB3",
        "colab_type": "text"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_FWQcjFjWNww",
        "colab_type": "text"
      },
      "source": [
        "6. Predict the categories for articles in the given test data set (\"diseases-test-without-labels.csv\"). Save the predictions into \"diseases-test-preds.csv\" (it should have the exact same format as the \"diseases-train.csv\" file) and upload along with the completed ipynb file. We will use your generated \"diseases-test-preds.csv\" file for evaluting the final perforamnce of your code (using macro-averaged F1). \n",
        "\n",
        "The three best performing submissions on test data will get **bonus points** (5% of the assignment grade for the 1st place, 3% for 2nd, and 2% for 3rd). The winner will be announced in the class after final evaluation."
      ]
    }
  ]
}