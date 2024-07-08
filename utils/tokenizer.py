
def Kmers_funct(seq, size=6):
    """
    Function that takes in a sequence and returns a list of kmers
    Parameters
    ----------
    seq : str
        DNA sequence
    size : int
        Size of kmers to return
    Returns
    -------
    list
        List of kmers
    """
    return [seq[x:x+size].upper() for x in range(len(seq) - size + 1)]

def tok_func(x): 
    """
    Function that takes in a sequence and returns a joined string of kmers
    Parameters
    ----------
    x : str
        DNA sequence
    Returns
    -------
    str
        Joined string of kmers
    """
    return " ".join(Kmers_funct(x))