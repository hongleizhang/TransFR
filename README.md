
## Introduction

**TransFR: Transferable Federated Recommendation with Pre-trained Language Models**


Federated recommendations (FRs), facilitating multiple local clients to collectively learn a global model without disclosing user private data, have emerged as a prevalent architecture for privacy-preserving recommendations. In conventional FRs, a dominant paradigm is to utilize discrete identities to represent users/clients and items, which are subsequently mapped to domain-specific embeddings to participate in model training. Despite considerable performance, we reveal three inherent limitations that can not be ignored in federated settings, i.e., non-transferability across domains, unavailability in cold-start settings, and potential privacy violations during federated training. To this end, we propose a transferable federated recommendation model aiming to recommend cross-domain items for shared clients, TransFR, which delicately incorporates the general capabilities empowered by pre-trained models and the personalized abilities by fine-tuning local private data. Specifically, it first learns domain-agnostic representations of items by exploiting pre-trained models with public textual corpora. To tailor for federated recommendation, we further introduce an efficient federated fine-tuning and a local training mechanism. This facilitates personalized local heads for each client by utilizing their private behavior data. By incorporating pre-training and fine-tuning within FRs, it greatly improves the adaptation efficiency transferring to a new domain and the generalization capacity to address cold-start issues. Through extensive experiments on several datasets, we demonstrate that our TransFR model surpasses several state-of-the-art FRs in terms of accuracy, transferability, and privacy.

## Requirements

The code is built on `Python=3.8` and `Pytorch=1.3`.

The other necessary Python libraries are as follows:

* numpy==1.21.5
* pandas==1.3.5
* psutil==5.9.6
* scikit_learn==1.0.2
* torch==2.1.1
* tqdm==4.65.2
* transformers==4.36.2

To install these, please run the following commands:

  `pip install -r requirements.txt`
  

## Quick Start

Please change the used dataset and hyperparameters in `options.py`.

To run TransFR on Amazon dataset:

  `nohup python -u main.py > transfr_main_music_book.out 2>&1 &`

