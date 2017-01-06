import sys
import os
import ConfigParser
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

cf = ConfigParser.ConfigParser()
cf.read("test.conf")
secs = cf.sections()
print 'sections:', secs
opts = cf.options("sec_a")
print 'options:', opts
kvs = cf.items("sec_a")
print 'sec_a:', kvs

str_val = cf.get("sec_a", "a_key1")
int_val = cf.getint("sec_a", "a_key2")

print "value for sec_a's a_key1:", str_val, type(str_val)
print "value for sec_a's a_key2:", int_val, type(int_val)

print cf.get("sec_b","b_key3")

