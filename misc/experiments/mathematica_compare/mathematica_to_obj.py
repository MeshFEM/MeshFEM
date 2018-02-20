import sys, re
coords,elms = sys.argv[1:]

for cl in file(coords):
    m = re.match('{([^,]+), ([^}]*)}', cl)
    if (not m): raise Exception("Fail.")
    print "v %s %s" % (m.group(1), m.group(2))
for el in file(elms):
    m = re.match('{([^,]+), ([^,]*), ([^,]*), .*}', el)
    if (not m): raise Exception("Fail.")
    print "f ", " ".join(m.groups())
