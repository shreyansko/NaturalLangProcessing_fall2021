

my_str1 = "this is just a test"

#clean = re.sub('[^A-Za-z0-9]+', " ", my_str).strip()

#print(clean)

# accessing a function defined in a different .py notebook
#  * imports all, if you want to import specific ones, refer to them by name
from utils.my_functions import *
clean_test(my_str1)
    
the_path = "/Users/shreyanskothari/Desktop/Natural Language Processing/corpuses"

file_seeker(the_path)

def file_seeker1(path_in):
    import os
    import pandas as pd
    my_df = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(path_in):
       tmp_dir_name = dirName.split("/")[-1::][0]
       try:
            for word in fileList:
                my_df = my_df.append({"label": tmp_dir_name,
                                      "path": dirName+"/"+word}, ignore_index =True)
       except Exception as e:
            print (e)
            pass
    return my_df

def file_seeker2(path_in):
    import os
    import pandas as pd
    import re
    my_df = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(path_in):
       tmp_dir_name = dirName.split("/")[-1::][0]
       try:
            for word in fileList:
                f = open(dirName+"/"+word, "r")
                read = f.read()
                clean = clean_test(read)
                f.close()
                #re.sub('[^A-Za-z0-9]+', " ", f).strip().lower()
                my_df = my_df.append({"label": tmp_dir_name,
                                      "body": clean.lower()}, ignore_index =True)
       except Exception as e:
            print (e)
            pass
    return my_df




my_pd1 = file_seeker2(the_path)