# Deep Sort 

This is the implemention of deep sort with pytorch, referenced from https://github.com/ZQPei/deep_sort_pytorch/tree/master/deep_sort.

We did the following changes: 
1) The original DeepSort is only for a single type of object, we add the object type into the sort/track.py, created a new file mytrack.py and switched the mytrack.py in mytracker.py, and changed _initiate_track in mytracker.py. Create mydetection.py file. Changed the update function of DeepSort, created a new class: MyDeepSort and new file mydeep_sort.py