def estimate_mean(n, delta):
    sample = np.loadtxt("data.txt",skiprows=3)
    i = 0.0
    count_groups = 0
    bin_groups = []
    d3 = dict()
    bin_sum = {}
    
    a = np.arange(sample.size)//n
    b = np.bincount(np.arange(a.size)//n,sample)/np.bincount(np.arange(a.size)//n)
    sampling_means = b.tolist()
    
    for samples in sample:
        while i < 1.0 and count_groups < (1/delta):
            bin_groups.append([i, round(i + delta, 10)])
            i = round(i + delta, 5)
            count_groups += 1
            
    for k in bin_groups:
        bin_dict = { k[0] :[] for k in bin_groups }
        bin_pmf_dict = { k[0] :[] for k in bin_groups }
        
        
    mean_count = len(sampling_means)
    
    while mean_count > 0:
        for p in bin_dict:
            for means in sampling_means:
                if means > p and means < p+delta:
                    bin_dict.setdefault(p,[]).append(means)
                    mean_count = mean_count -1

    for p in bin_dict:
        bin_pmf_dict[p] = len(bin_dict[p])/(100000/n)

    mean_PMF = sum(k*v for k,v in bin_pmf_dict.items())
    
    return bin_pmf_dict, mean_PMF