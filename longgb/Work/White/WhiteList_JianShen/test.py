from fp_growth import find_frequent_itemsets
for itemset in find_frequent_itemsets(transactions, minsup):
    print itemset