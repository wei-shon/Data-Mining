"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/a
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python apriori.py -f DATASET.csv -s minSupport

    $python apriori.py -f DATASET.csv -s 0.15
"""

import sys
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    _shouldBePurning = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)
        if support >= minSupport:
            _itemSet.add(item)
        else :
            _shouldBePurning.add(item)

    return _itemSet , _shouldBePurning

# 得到itemset的組合
def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
            
    return itemSet, transactionList

def purning(currentLSet , _shouldBePurningSet):
    for items in currentLSet:
        for purning_items in _shouldBePurningSet:
            if items.issubset(purning_items):
                currentLSet.remove(items)
                print("purning item : ",items)
                continue
    return currentLSet
    
def closed_frequent_count(close_frequent_itemset_list,freqSet):
    close_frequent_itemset=[]
    for k in range(len(close_frequent_itemset_list)):
        if k == len(close_frequent_itemset_list)-1:
            for item in close_frequent_itemset_list[k]:
                close_frequent_itemset.append(item)
            break
        else:
             
            for item_sub in close_frequent_itemset_list[k]:
                for item in close_frequent_itemset_list[k+1]:
                    if item.issubset(item_sub):
                        if freqSet[item_sub]> freqSet[item]:
                            close_frequent_itemset.append(item_sub)
    # print(close_frequent_itemset)
    return close_frequent_itemset

def runApriori(data_iter, minSupport):
    start_time = time.time()
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
    """
    beforePunning = []
    afterPunning = []
    close_frequent_itemset_list = []
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    oneCSet , _shouldBePurningSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    
    currentLSet = oneCSet

    # 記得第一次purning 結果，其實根本沒有purning
    beforePunning.append(len(currentLSet))
    afterPunning.append(len(currentLSet))

    close_frequent_itemset_list.append(currentLSet)
    k = 2
    while currentLSet != set([]):    
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)

        # 記得第k次purning 結果
        beforePunning.append(len(currentLSet))
        currentLSet = purning(currentLSet , _shouldBePurningSet)
        afterPunning.append(len(currentLSet))

        currentCSet , _shouldBePurningSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        currentLSet = currentCSet
        #先記得目前的frequent item set才能做後面的closed frequent item set
        if currentLSet != set([]) : 
            close_frequent_itemset_list.append(currentLSet)
        k = k + 1

    task1_end_time = time.time()
    # do closed frequent item set
    close_frequent_itemset = closed_frequent_count(close_frequent_itemset_list,freqSet)
    task2_end_time = time.time()

    # take the task time
    task1_time = task1_end_time - start_time
    task2_time = task2_end_time - start_time

    print("computation time Task1 : {:f}s".format(task1_time))
    print("computation time Task2 : {:f}s".format(task2_time))
    print("ratio of computation time : {:f}%".format((task2_time/task1_time)*100 ))

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems , beforePunning , afterPunning , close_frequent_itemset , freqSet , transactionList


def printResults(items):
    """prints the generated itemsets sorted by support """
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))


def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.3f" % (str(item), support)
        i.append(x)
    return i


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record

def result1(items , fd):
    frequence_itemset_counter=0
    for item, support in sorted(items, key=lambda x: x[1])[::-1]:
        fd.write("{:.1f}%\t{{".format(support*100))
        for i in range(len(item)):
            if i != len(item)-1:
                fd.write("{:s,}".format(item[i]) )
            else:
                fd.write("{:s}".format(item[i]) )
        fd.write("}\n")
        frequence_itemset_counter+=1
    return frequence_itemset_counter

def result2(frequence_itemset_counter , beforePunning , afterPunning , fd2):
    fd2.write(str(frequence_itemset_counter)+'\n')
    k = 1
    for i in range(len(beforePunning)):
        fd2.write("{:d}\t{:d}\t{:d}\n".format( k , beforePunning[i] , afterPunning[i]))
        k+=1

def task2(close_frequent_itemset , freqSet , transactionList , fd3):
    fd3.write(str(len(close_frequent_itemset))+'\n')
    for i in range(len(close_frequent_itemset)):
        for j in range(len(close_frequent_itemset)):
            if freqSet[close_frequent_itemset[i]]> freqSet[close_frequent_itemset[j]]:
                temp = close_frequent_itemset[i]
                close_frequent_itemset[i] = close_frequent_itemset [j]
                close_frequent_itemset[j] = temp
    for i in range(len(close_frequent_itemset)):
        support = float(freqSet[close_frequent_itemset[i]])/len(transactionList)
        fd3.write("{:.1f}%\t{{".format(support*100))
        # print(type(iter(close_frequent_itemset[i])))
        # # print(close_frequent_itemset[i][0])
        for id,item in enumerate(close_frequent_itemset[i]):
            if id != len(close_frequent_itemset[i])-1:
                fd3.write("{:},".format(item))
            else:
                fd3.write("{:}".format(item))
        fd3.write('}\n')

if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option(
        "-f", "--inputFile", dest="input", help="filename containing csv", default='A.csv'
    )
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.1,
        type="float",
    )
    
    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS

    items , beforePunning , afterPunning , close_frequent_itemset , freqSet , transactionList = runApriori(inFile, minSupport)

    # printResults(items)
     # open file
    fd1 = open('step2_task1_{:s}_{:.1f}%_result1.txt'.format(options.input[:-4],minSupport*100),"w")
    fd2 = open('step2_task1_{:s}_{:.1f}%_result2.txt'.format(options.input[:-4],minSupport*100),"w")
    fd3 = open('step2_task2_{:s}_{:.1f}%_result1.txt'.format(options.input[:-4],minSupport*100),"w")
    #wrete the result1
    frequence_itemset_counter = result1(items,fd1) 
    #write the result 2
    result2(frequence_itemset_counter , beforePunning , afterPunning , fd2)
    task2(close_frequent_itemset , freqSet , transactionList , fd3)
    
    fd1.close()
    fd2.close()
    fd3.close()