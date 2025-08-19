"""Generates data for train/test algorithms"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pickle
import os
import random
import tldextract
import numpy as np
import csv

import banjori, corebot, cryptolocker, dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda, matsnu, suppobox, gozi

# 데이터/리소스는 이 파일 위치 기준으로
DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / 'traindata.pkl'

def get_alexa(num: int, filename: str | Path | None = None):
    """Grabs Alexa 1M"""
    if filename is None:
        filename = DATA_DIR / "top-1m.csv"  # 파일 위치 기준
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"top-1m.csv not found at {filename.resolve()}")

    domains = []
    with filename.open('r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i >= num:
                break
            domain = tldextract.extract(row[1]).domain
            domains.append(domain)
    return domains

def gen_malicious(num_per_dga=10000):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []

    # banjori
    banjori_seeds = ['somestring','firetruck','bulldozer','airplane','racecar','apartment','laptop','laptopcomp',
                     'malwareisbad','crazytrain','thepolice','fivemonkeys','hockey','football','baseball',
                     'basketball','trackandfield','fieldhockey','softball','redferrari','blackcheverolet',
                     'yellowelcamino','blueporsche','redfordf150','purplebmw330i','subarulegacy','hondacivic',
                     'toyotaprius','sidewalk','pavement','stopsign','trafficlight','turnlane','passinglane',
                     'trafficjam','airport','runway','baggageclaim','passengerjet','delta1008','american765',
                     'united8765','southwest3456','albuquerque','sanfrancisco','sandiego','losangeles',
                     'newyork','atlanta','portland','seattle','washingtondc']
    segs_size = max(1, int(num_per_dga/len(banjori_seeds)))
    for seed in banjori_seeds:
        domains += banjori.generate_domains(segs_size, seed)
        labels += ['banjori']*segs_size

    domains += corebot.generate_domains(num_per_dga); labels += ['corebot']*num_per_dga

    # cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = max(1, int(num_per_dga/len(crypto_lengths)))
    for L in crypto_lengths:
        domains += cryptolocker.generate_domains(segs_size, seed_num=random.randint(1, 1_000_000), length=L)
        labels += ['cryptolocker']*segs_size

    domains += dircrypt.generate_domains(num_per_dga); labels += ['dircrypt']*num_per_dga

    # kraken
    half = max(1, num_per_dga//2)
    domains += kraken.generate_domains(half, datetime(2016,1,1), 'a', 3); labels += ['kraken']*half
    domains += kraken.generate_domains(half, datetime(2016,1,1), 'b', 3); labels += ['kraken']*half

    # locky
    per = max(1, int(num_per_dga/11))
    for i in range(1, 12):
        domains += lockyv2.generate_domains(per, config=i)
        labels += ['locky']*per

    # pykspa
    domains += pykspa.generate_domains(num_per_dga, datetime(2016,1,1)); labels += ['pykspa']*num_per_dga

    # qakbot
    domains += qakbot.generate_domains(num_per_dga, tlds=[]); labels += ['qakbot']*num_per_dga

    # ramdo
    ramdo_lengths = range(8, 32)
    segs_size = max(1, int(num_per_dga/len(ramdo_lengths)))
    for L in ramdo_lengths:
        domains += ramdo.generate_domains(segs_size, seed_num=random.randint(1,1_000_000), length=L)
        labels += ['ramdo']*segs_size

    # ramnit
    domains += ramnit.generate_domains(num_per_dga, 0x123abc12); labels += ['ramnit']*num_per_dga

    # simda
    simda_lengths = range(8, 32)
    segs_size = max(1, int(num_per_dga/len(simda_lengths)))
    for L in simda_lengths:
        domains += simda.generate_domains(segs_size, length=L, tld=None, base=random.randint(2, 2**32))
        labels += ['simda']*segs_size

    # matsnu / suppobox / gozi
    domains += matsnu.generate_domains(num_per_dga, include_tld=False); labels += ['matsnu']*num_per_dga
    domains += suppobox.generate_domains(num_per_dga, include_tld=False); labels += ['suppobox']*num_per_dga
    domains += gozi.generate_domains(num_per_dga, include_tld=False); labels += ['gozi']*num_per_dga

    return domains, labels

def gen_data(force=False):
    """Grab all data for train/test and save"""
    if force or (not Path(DATA_FILE).exists()):
        domains, labels = gen_malicious(10000)

        # 동일 개수 benign 추가
        num_benign = len(domains)
        domains += get_alexa(num_benign)  # filename 기본값: DATA_DIR/top-1m.csv
        labels  += ['benign'] * num_benign

        with open(DATA_FILE, 'wb') as f:
            pickle.dump(list(zip(labels, domains)), f)

def get_data(force=False):
    """Returns data and labels"""
    gen_data(force)
    with open(DATA_FILE, 'rb') as f:
        return pickle.load(f)

def get_malware_labels(labels):
    malware_labels = sorted(set(labels) - {'benign'})
    return malware_labels

def expand_labels(labels):
    """멀웨어 패밀리별 one-vs-rest 0/1 라벨 시퀀스 리스트 반환"""
    all_Ys = []
    for mal in get_malware_labels(labels):
        y = [1 if label == mal else 0 for label in labels]
        all_Ys.append(y)
    return all_Ys

def get_labels():
    return ['main','corebot','dircrypt','kraken','pykspa','qakbot','ramnit','locky',
            'banjori','cryptolocker','ramdo','simda','matsnu','suppobox','gozi']

def get_losses():
    return {k: 'binary_crossentropy' for k in get_labels()}

def get_loss_weights():
    w = {k: 0.1 for k in get_labels()}
    w['main'] = 1.0
    return w

def y_list_to_dict(all_Ys):
    return dict((label, np.array(y)) for label, y in zip(get_labels(), all_Ys))
