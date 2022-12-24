import pandas as pd

def get_split_points(filename, skiprows=0, testrows=10, sep=' '):
    '''
    Find the column split points in a fixed width file
    '''
    with open(filename) as myfile:
        head = [next(myfile) for x in range(skiprows)]
        testset = [next(myfile) for x in range(testrows)]
        maxlength = len(max(testset , key = len))
        spacecounts = [0 for col in range(maxlength)]
        for string in testset:
            for i in range(len(string)):
                if string[i] == sep:
                    spacecounts[i] += 1
        indices = [i for i, x in enumerate(spacecounts) if x == testrows]
        split_points = [0] + [x for x in indices if x-1 not in indices and x > 0 ]
        return split_points
        
def parse_fixedwidth_file(filename, skiprows=0, headers=True,
                          testrows=10, defaulttype=str, datatypes={}, sep=' '):
    '''
    Parse a fixed width file and return a list of dictionaries with the results
    '''
    split_points = get_split_points(filename, skiprows, testrows=testrows, sep=sep)
    with open(filename) as f:
        # Skip top rows
        _ = [next(f) for x in range(skiprows)]
        # Determine column names
        if headers:
            headerline = next(f)
            names = [headerline[i:j].strip() for i,j in zip(split_points, split_points[1:] + [None])]
        else:
            names = ['Column_' + str(i) for i,j in zip(split_points, split_points[1:] + [None])]
        # Read data rows into a list of dictionaries
        data = []
        while True:
            line = f.readline().replace(sep, ' ')
            if not line:
                break
            values = [line[i:j].strip() for i,j in zip(split_points, split_points[1:] + [None])]
            data.append({k:v for k,v in zip(names,values)})
    # Convert data types
    data = [{key : (datatypes[key](val) if (len(val) > 1 and key in datatypes.keys()) 
                                        else (defaulttype(val) if len(val) > 0 else val))
                    for key, val in sub.items()}
                    for sub in data]
        
    return  data

def parse_fwf_to_df(filename, skiprows=0, headers=True,
                          testrows=10, defaulttype=str, datatypes={}, sep=' '):
    data = parse_fixedwidth_file(filename, skiprows, headers, testrows, defaulttype, datatypes)
    return pd.DataFrame(data)
