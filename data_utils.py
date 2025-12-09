#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

import random
from io import open
import os
import numpy as np
class DataUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def rename(self, datafile):
        """
        Distinguish two types of node and rename
        """
        with open(os.path.join(self.model_path,"edges.dat"), "w") as fw:
            with open(datafile, "r", encoding="UTF-8") as fin:
                line = fin.readline()
                while line:
                    user, item, rating = line.strip().split("\t")
                    fw.write("u"+ user + "\t" + "i" + item + "\t" + rating + "\n")
                    line = fin.readline()

    def split_data(self, percent):
        """
        split data
        :param percent:
        :return:
        """
        test_user,test_item,test_rate,rating, data = set(), set(), {},{}, []
        with open(os.path.join(self.model_path, "edges.dat"), "r") as fin, open(os.path.join(self.model_path, "edges_train.dat"),"w") as ftrain, open(os.path.join(self.model_path,"edges_test.dat"), "w") as ftest:
            next(fin)
            for line in fin.readlines():
                user, item, rate = line.strip().split("\t")
                if rating.get(user) is None:
                    rating[user] = {}
                rating[user][item] = rate
                data.append([user, item, rate])
            np.random.shuffle(data)
            split_index = int(len(data) * percent)
            train_data = data
            test_data = data[split_index:]
            for user, item, rate in train_data:
                ftrain.write("u"+ user + "\t" + "i"+ item + "\t" + rating[user][item] + "\n")
            for user, item, rate in test_data:
                if test_rate.get(user) is None:
                    test_rate[user] = {}
                test_rate[user][item] = float(rating[user][item])
                test_user.add(user)
                test_item.add(item)
                ftest.write("u"+ user + "\t" + "i"+ item + "\t" + rating[user][item] + "\n")
        return test_user, test_item, test_rate

    def read_data(self,filename=None):
        if filename is None:
            filename = os.path.join(self.model_path,"edges_test.dat")
        users,items,rates = set(), set(), {}
        with open(filename, "r", encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rate = line.strip().split()
                if rates.get(user) is None:
                    rates[user] = {}
                rates[user][item] = float(rate)
                users.add(user)
                items.add(item)
                line = fin.readline()
        return users, items, rates



