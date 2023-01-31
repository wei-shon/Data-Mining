class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count              # support
        self.parent = parent
        self.next = None               # the same elements
        self.children = {}

    def display(self, ind=1):
        print(''*ind, self.item, '', self.count)
        for child in self.children.values():
            child.display(ind+1)

class FPgrowth:
    def __init__(self,len_Transaction, min_support=3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.len_Transaction = len_Transaction
    '''
    Function:  transfer2FrozenDataSet
    Description: transfer data to frozenset type
    Input:  data              dataType: ndarray     description: train_data
    Output: frozen_data       dataType: frozenset   description: train_data in frozenset type
    '''
    def transfer2FrozenDataSet(self, data):
        frozen_data = list()
        for elem in data:
            frozen_data.append(frozenset(elem))
        # frozen_data = {}
        # for elem in data:
        #     frozen_data[frozenset(elem)] = 1

        # fd = open("a.txt","w")
        # for item in frozen_data:
        #     for i in item:
        #         fd.write(i+',')
        # fd.close
        return frozen_data

    '''
      Function:  updataTree
      Description: updata FP tree
      Input:  data              dataType: ndarray     description: ordered frequent items
              FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
              count             dataType: count       description: the number of a record 
    '''
    def updataTree(self, data, FP_tree, header, count):
        frequent_item = data[0]
        if frequent_item in FP_tree.children:
            FP_tree.children[frequent_item].count += count
        else:
            FP_tree.children[frequent_item] = FPNode(frequent_item, count, FP_tree)
            if header[frequent_item][1] is None:
                header[frequent_item][1] = FP_tree.children[frequent_item]
            else:
                self.updateHeader(header[frequent_item][1], FP_tree.children[frequent_item]) # share the same path

        if len(data) > 1:
            # print('done')
            self.updataTree(data[1::], FP_tree.children[frequent_item], header, count)  # recurrently update FP tree

    '''
      Function: updateHeader
      Description: update header, add tail_node to the current last node of frequent_item
      Input:  head_node           dataType: FPNode     description: first node in header
              tail_node           dataType: FPNode     description: node need to be added
    '''
    def updateHeader(self, head_node, tail_node):
        while head_node.next is not None:
            head_node = head_node.next
        head_node.next = tail_node

    '''
      Function:  createFPTree
      Description: create FP tree
      Input:  train_data        dataType: ndarray     description: features
      Output: FP_tree           dataType: FPNode      description: FP tree
              header            dataType: dict        description: header pointer table
    '''
    def createFPTree(self, train_data):
        initial_header = {}
        # 1. the first scan, get singleton set
        for record in train_data:
            for item in record:
                initial_header[item] = initial_header.get(item, 0) + 1

        # get singleton set whose support is large than min_support. If there is no set meeting the condition,  return none
        header = {}
        for k in initial_header.keys():
            if initial_header[k] / self.len_Transaction >= self.min_support:
                # print( k , " : " , initial_header[k]/len(train_data)*100) # 測試 k=1 的 frequent itemset 是否正確
                header[k] = initial_header[k]
        frequent_set = set(header.keys())
        if len(frequent_set) == 0:
            return None, None

        # enlarge the value, add a pointer
        for k in header:
            header[k] = [header[k], None]
        # 2. the second scan, create FP tree
        FP_tree = FPNode('root', 1, None)        # root node
        for record in train_data:
            frequent_item = {}
            for item in record:                # if item is a frequent set， add it
                if item in frequent_set:       # 2.1 filter infrequent_item
                    frequent_item[item] = header[item][0]

            if len(frequent_item) > 0:
                ordered_frequent_item = [val[0] for val in sorted(frequent_item.items(), key=lambda val:val[1], reverse=True)]  # 2.1 sort all the elements in descending order according to count
                self.updataTree(ordered_frequent_item, FP_tree, header, 1) # 2.2 insert frequent_item in FP-Tree， share the path with the same prefix
        return FP_tree, header

    '''
      Function: ascendTree
      Description: ascend tree from leaf node to root node according to path
      Input:  node           dataType: FPNode     description: leaf node
      Output: prefix_path    dataType: list       description: prefix path
              
    '''
    def ascendTree(self, node):
        prefix_path = []
        while node.parent != None and node.parent.item != 'root':
            node = node.parent
            prefix_path.append(node.item)
        return prefix_path

    '''
    Function: getPrefixPath
    Description: get prefix path
    Input:  base          dataType: FPNode     description: pattern base
            header        dataType: dict       description: header
    Output: prefix_path   dataType: dict       description: prefix_path
    '''
    def getPrefixPath(self, base, header):
        prefix_path = {}
        start_node = header[base][1]
        prefixs = self.ascendTree(start_node)
        if len(prefixs) != 0:
            prefix_path[frozenset(prefixs)] = start_node.count

        while start_node.next is not None:
            start_node = start_node.next
            prefixs = self.ascendTree(start_node)
            if len(prefixs) != 0:
                prefix_path[frozenset(prefixs)] = start_node.count
        return prefix_path

    '''
    Function: findFrequentItem
    Description: find frequent item
    Input:  header               dataType: dict       description: header [name : (count, pointer)]
            prefix               dataType: dict       description: prefix path
            frequent_set         dataType: set        description: frequent set
    '''
    def findFrequentItem(self, header, prefix, frequent_set):
        # for each item in header, then iterate until there is only one element in conditional fptree
        if not header: # 如果遇到header已經是空的，代不用在往下做了。 type(header) = dict
            return
        header_items = [val[0] for val in sorted(header.items(), key=lambda val: val[1][0])]
        if len(header_items) == 0:
            return
        # print(header_items)
        for base in header_items:
            # print(header[base][0])
            new_prefix = prefix.copy()
            new_prefix.add(base)
            support = header[base][0]
            frequent_set[frozenset(new_prefix)] = support

            prefix_path = self.getPrefixPath(base, header)
            if len(prefix_path) != 0:
                conditonal_tree, conditional_header = self.createFPTree(prefix_path)
                if conditional_header is not None:
                    self.findFrequentItem(conditional_header, new_prefix, frequent_set)

    '''
     Function:  generateRules
     Description: generate association rules
     Input:  frequent_set       dataType: set         description:  current frequent item
             rule               dataType: dict        description:  an item in current frequent item
     '''
    def generateRules(self, frequent_set, rules):
        for frequent_item in frequent_set:
            if len(frequent_item) > 1:
                self.getRules(frequent_item, frequent_item, frequent_set, rules)

    '''
     Function:  removeItem
     Description: remove item
     Input:  current_item       dataType: set         description:  one record of frequent_set
             item               dataType: dict        description:  support_degree 
     '''
    def removeItem(self, current_item, item):
        tempSet = []
        for elem in current_item:
            if elem != item:
                tempSet.append(elem)
        tempFrozenSet = frozenset(tempSet)
        return tempFrozenSet

    '''
     Function:  getRules
     Description: get association rules
     Input:  frequent_set       dataType: set         description:  one record of frequent_set
             rule               dataType: dict        description:  support_degree 
     '''
    def getRules(self, frequent_item, current_item, frequent_set, rules):
        for item in current_item:
            subset = self.removeItem(current_item, item)
            confidence = frequent_set[frequent_item]/frequent_set[subset]
            if confidence >= self.min_confidence:
                flag = False
                for rule in rules:
                    if (rule[0] == subset) and (rule[1] == frequent_item - subset):
                        flag = True

                if flag == False:
                    rules.append((subset, frequent_item - subset, confidence))

                if (len(subset) >= 2):
                    self.getRules(frequent_item, subset, frequent_set, rules)

    '''
      Function:  train
      Description: train the model
      Input:  train_data       dataType: ndarray   description: items
              display          dataType: bool      description: print the rules
      Output: rules            dataType: list      description: the learned rules
              frequent_items   dataType: list      description: frequent items set
    '''
    def train(self, data, display=True):
        data = self.transfer2FrozenDataSet(data)
        FP_tree, header = self.createFPTree(data)
        #FP_tree.display()
        frequent_set = {}
        prefix_path = set([])
        self.findFrequentItem(header, prefix_path, frequent_set)
        # rules = []
        # self.generateRules(frequent_set, rules)

        # if display:
        #     print("Frequent Items:")
        #     for item in frequent_set:
        #         print(item)
        #     print("_______________________________________")
        #     print("Association Rules:")
        #     for rule in rules:
        #         print(rule)
        return frequent_set

import sys
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser

def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split(","))
            yield record
def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = record
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
            
    return itemSet, transactionList
def step3_result1(frequent_set,transactionList ,fd1):
    frequence_itemset_counter=0
    for items , value in sorted(frequent_set.items() , key=lambda x:x[1], reverse=True):
        support = float(value)/len(transactionList)
        fd1.write("{:.1f}%\t{{".format(support*100))
        for id , item in enumerate(items):
            if id != len(items)-1:
                fd1.write("{:},".format(item))
            else:
                fd1.write("{:}".format(item))
        fd1.write('}\n')
        frequence_itemset_counter+=1
    return frequence_itemset_counter

def step3_result2(frequence_itemset_counter , fd2):
    fd2.write(str(frequence_itemset_counter)+'\n')


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

    itemset , transactionList = getItemSetTransactionList(inFile)

 

    FP = FPgrowth(len(transactionList),minSupport)
    
    start_time = time.time()
    frequent_set = FP.train(transactionList)
    end_time = time.time()
    use_time = end_time - start_time
    print("computation time Step3 Task1 : {:f}s".format(use_time))
    # print(frequent_set)
    
    # # # open file
    fd1 = open('step3_task1_{:s}_{:.1f}%_result1.txt'.format(options.input[:-4],minSupport*100),"w")
    fd2 = open('step3_task1_{:s}_{:.1f}%_result2.txt'.format(options.input[:-4],minSupport*100),"w")

    #wrete the result1
    frequence_itemset_counter = step3_result1(frequent_set ,transactionList,fd1) 
    #write the result 2
    step3_result2(frequence_itemset_counter , fd2)

    fd1.close()
    fd2.close()


    # for i in range(len(frequent_set)):
    #     support = float(freqSet[frequent_set[i]])/len(transactionList)
    #     # print(type(iter(close_frequent_itemset[i])))
    #     # # print(close_frequent_itemset[i][0])
    #     for id,item in enumerate(close_frequent_itemset[i]):
    #         if id != len(close_frequent_itemset[i])-1:
    #             fd3.write("{:},".format(item))
    #         else:
    #             fd3.write("{:}".format(item))
    #     fd3.write('}\n')
    # print(type(frequent_set))
    
    