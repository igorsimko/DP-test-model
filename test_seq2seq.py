import sys
import json
from rouge import *

hyp = "Religious Business groups"
ref = "Animal groups equipment"

print(rouge([hyp], [ref]))