class Binning:
    
    def __init__(self, c, r_ip1, N, N_bins, lags, store_frame_rate=1, uniform_bins = True, min_count = 0, verbose = True):
        
        #number of conditional variables
        N_c = c.shape[1]
        
        self.N_c = N_c
        self.N = N
        self.r_ip1 = r_ip1
        self.c = c
        self.N_bins = N_bins
        self.N_s = r_ip1.size/N**2
        self.lags = lags
        self.max_lag = np.max(lags)*store_frame_rate

        self.covar = {}
        
        for i in range(N_c):
            #self.covar[i] = np.zeros([N**2, max_lag])
            self.covar[i] = []

        #compute bins of c_i
        if uniform_bins == True:
            bins = self.get_bins(N_bins)
        else:
            bins = self.get_bins_same_nsamples(N_bins)
        
        #count = numver of r_ip1 samples in each bin
        #binedges = same as bins
        #binnumber = bin indices of the r_ip1 samples. A 1D array, no matter N_c
        count, binedges, binnumber = stats.binned_statistic_dd(c, r_ip1, statistic='count', bins=bins)
        #x_idx = np.unravel_index(binnumber, [len(b) + 1 for b in self.bins])
        #x_idx = [x_idx[i] - 1 for i in range(N_c)]
        
        #the unique set of binnumers which have at least one r_ip1 sample
        unique_binnumbers = np.unique(binnumber)
        
        #some scalars
        binnumber_max = np.max(unique_binnumbers)
        
        #array containing r_ip1 indices sorted PER BIN
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #SHOULD BE A 1D ARRAY, AN ARRAY OF SIZE [MAX(BINNUMBER), MAX(COUNT)]
        #WILL STORE MOSTLY ZEROS IN HIGHER DIMENSIONS, LEADING TO MEMORY FAILURES
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        idx_of_bin = []
        
        #number of samples in each bin
        size_of_bin = []
        
        for b in unique_binnumbers:
            #r_ip1 indices of bin b
            tmp = np.where(binnumber == b)[0]
            
            idx_of_bin.append(tmp)
            size_of_bin.append(tmp.size)

        #make a 1d array of the nested arrays
        idx_of_bin = np.array(list(chain(*idx_of_bin)))
        size_of_bin = np.array(size_of_bin)
        
        #the starting offset of each bin in the 1D array idx_of_bin
        #offset[unique_binnumbers] will give the starting index in idex_of_bin for
        #each bin. Some entries are zero, which correspond to empty bins (except for
        #the lowest non-empty binnumber)
        offset = np.zeros(unique_binnumbers.size).astype('int')
        offset[1:] = np.cumsum(size_of_bin[0:-1])
        tmp = np.zeros(binnumber_max+1).astype('int')
        tmp[unique_binnumbers] = offset
        self.offset = tmp
        
        ####################
        #empty bin handling#
        ####################
        
        #find indices of empty bins
        #Note: if 'full' is defined as 1 or more sample, binnumbers_nonempty is 
        #the same as binnumbers_unique
        x_idx_nonempty = np.where(count > min_count)
        x_idx_nonempty_p1 = [x_idx_nonempty[i] + 1 for i in range(N_c)]
        binnumbers_nonempty = np.ravel_multi_index(x_idx_nonempty_p1, [len(b) + 1 for b in bins]) 
        N_nonempty = binnumbers_nonempty.size
        
        #mid points of the bins
        x_mid = [0.5*(bins[i][1:] + bins[i][0:-1]) for i in range(N_c)]

        #mid points of the non-empty bins
        midpoints = np.zeros([N_nonempty, N_c])
        for i in range(N_c):
            midpoints[:, i] = x_mid[i][x_idx_nonempty[i]]

        
        self.verbose = verbose
        self.bins = bins
        self.count = count
        self.binnumber = binnumber
        self.binnumbers_nonempty = binnumbers_nonempty
        self.midpoints = midpoints
        self.idx_of_bin = idx_of_bin
        self.unique_binnumbers = unique_binnumbers
        self.compute_binnumber_per_bin()
        self.fill_in_blanks()

        #mean r / var per cell
        self.rmean, _, _ = stats.binned_statistic_dd(c, r_ip1, statistic='mean', bins=bins)
        #self.rstd, _, _ = stats.binned_statistic_dd(c, r_ip1, statistic=std_per_bin, bins=bins)
        #self.rmax, _, _ = stats.binned_statistic_dd(c, r_ip1, statistic='max', bins=bins)
        #self.rmin, _, _ = stats.binned_statistic_dd(c, r_ip1, statistic='min', bins=bins)
        #self.compute_c_samples_per_bin()

    #check which c_i fall within empty bins and correct binnumbers_i by
    #projecting to the nearest non-empty bin
    def check_outliers(self, binnumbers_i, c_i):
        
        #find out how many BINS with outliers there are
        unique_binnumbers_i = np.unique(binnumbers_i)
        idx = np.where(np.in1d(unique_binnumbers_i, self.binnumbers_nonempty) == False)[0]
        N_outlier_bins = idx.size
        
        if N_outlier_bins > 0:
            
            #index of outlier SAMPLES in binnumbers_i
            outliers_idx = np.in1d(binnumbers_i, unique_binnumbers_i[idx]).nonzero()[0]
            N_outliers = outliers_idx.size
            
            if self.verbose == True:
                print N_outlier_bins, ' bins with', N_outliers ,'outlier samples found'
        
            #x location of outliers
            x_outliers = np.copy(c_i[outliers_idx])
        
            #find non-empty bin closest to the outliers
            closest_idx = np.zeros(N_outliers).astype('int')
            for i in range(N_outliers):
                if self.N_c == 1:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i], 2, 1)
                else:
                    dist = np.linalg.norm(self.midpoints - x_outliers[i,:], 2, 1)
                closest_idx[i] = np.argmin(dist)
             
            binnumbers_closest = self.binnumbers_nonempty[closest_idx]
            
            check = self.binnumbers_nonempty[closest_idx]
            x_idx_check = np.unravel_index(check, [len(b) + 1 for b in self.bins])
            x_idx_check = [x_idx_check[i] - 1 for i in range(self.N_c)]
            
            #overwrite outliers in binnumbers_i with nearest non-empty binnumber
            binnumbers_closest = self.binnumbers_nonempty[closest_idx]

            if self.verbose == True:
                print 'Moving', binnumbers_i[outliers_idx], '-->', binnumbers_closest

            binnumbers_i[outliers_idx] = binnumbers_closest
            
    def get_binnumbers_i(self, c_i):

        #find in which bins the c_i samples fall, shape: (N**2,)
        _, _, binnumbers_i = stats.binned_statistic_dd(c_i, np.zeros(self.N**2), bins=self.bins)
                
        return binnumbers_i

    #create an apriori mapping between every possible bin and the nearest 
    #non-empty bin. Non-empty bins will link to themselves.
    def fill_in_blanks(self):
        
        bins_padded = []
        
        #mid points of all possible bins, including ghost bins
        for i in range(self.N_c):
            dx1 = self.bins[i][1] - self.bins[i][0]
            dx2 = self.bins[i][-1] - self.bins[i][-2]
            
            #pad the beginning and end of current 1D bin with extrapolated values
            bin_pad = np.pad(self.bins[i], (1,1), 'constant', constant_values=(self.bins[i][0]-dx1, self.bins[i][-1]+dx2))    
            bins_padded.append(bin_pad)
        
        #compute the midpoints of the padded bins
        x_mid_pad = [0.5*(bins_padded[i][1:] + bins_padded[i][0:-1]) for i in range(self.N_c)]
        self.x_mid_pad_tensor = np.array(list(product(*x_mid_pad)))
        
        #total number bins
        self.max_binnumber = self.x_mid_pad_tensor.shape[0]
        
        mapping = np.zeros(self.max_binnumber).astype('int')
        
        for i in range(self.max_binnumber):
            
            #bin is nonempty, just use current idx
            if np.in1d(i, self.unique_binnumbers) == True:
                mapping[i] = i
            #bin is empty, find nearest non-empty bin
            else:
                binnumbers_i = np.array([i])
                self.check_outliers(binnumbers_i, self.x_mid_pad_tensor[i].reshape([1, self.N_c]))
                mapping[i] = binnumbers_i[0]
            
        self.mapping = mapping
        
    #visual representation of a 2D binning object. Also shows the mapping
    #between empty to nearest non-empty bins.
    def plot_2D_binning_object(self):

        if self.N_c != 2:
            print 'Only works for N_c = 2'
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=r'conditioning variable 1', ylabel=r'conditioning variable 2')
        
        #plot bins and (c1, c2) which corresponding to a r sample point
        ax.plot(self.c[:,0], self.c[:,1], '+', color='lightgray', alpha=0.3)
        ax.vlines(self.bins[0], np.min(self.c[:,1]), np.max(self.c[:,1]))
        ax.hlines(self.bins[1], np.min(self.c[:,0]), np.max(self.c[:,0]))
       
        ax.plot(self.x_mid_pad_tensor[:,0], self.x_mid_pad_tensor[:,1], 'g+')

        #plot the mapping
        for i in range(self.max_binnumber):
            ax.plot([self.x_mid_pad_tensor[i][0], self.x_mid_pad_tensor[self.mapping[i]][0]], \
                    [self.x_mid_pad_tensor[i][1], self.x_mid_pad_tensor[self.mapping[i]][1]], 'b', alpha=0.4)

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.show()
        
    #the data-driven model for the unresolved scales
    #Given c_i return r at time i+1 (r_ip1)
    def get_r_ip1(self, c_i, n_mc=1):
    
        #find in which bins the c_i samples fall
        _, _, binnumbers_i = stats.binned_statistic_dd(c_i, np.zeros(self.N**2), bins=self.bins)
        
        #dynamically corrects binnumbers_i if outliers are found
        #self.check_outliers(binnumbers_i, c_i)

        #static correction for outliers, using precomputed mapping array
        binnumbers_i = self.mapping[binnumbers_i]
        
        #self.check_outliers(binnumbers_i, c_i)

        #convert 1D binnumbers_i to equivalent ND indices
        x_idx = np.unravel_index(binnumbers_i, [len(b) + 1 for b in self.bins])
        x_idx = [x_idx[i] - 1 for i in range(self.N_c)]
        
        #random integers between 0 and max bin count for each index in binnumbers_i    
        I = np.floor(self.count[x_idx].reshape([self.N**2, 1])*np.random.rand(self.N**2, n_mc)).astype('int')
        
        #the correct offset for the 1D array idx_of_bin
        start = self.offset[binnumbers_i]
    
        #get random r sample from each bin indexed by binnumbers_i
        r = np.zeros([n_mc, self.N, self.N])
        
        for i in range(n_mc):
            r[i, :, :] = self.r_ip1[self.idx_of_bin[start + I[:, i]]].reshape([self.N, self.N])
        
        return np.mean(r, 0) #, binnumbers_i
    
    #the data-driven model for the unresolved scales
    #Given c_i return bin averaged r at time i+1 
    def get_mean_r_ip1(self, c_i):
        
        #find in which bins the c_i samples fall
        _, _, binnumbers_i = stats.binned_statistic_dd(c_i, np.zeros(self.N**2), bins=self.bins)
        
        #dynamically corrects binnumbers_i if outliers are found
        #self.check_outliers(binnumbers_i, c_i)

        #static correction for outliers, using precomputed mapping array
        binnumbers_i = self.mapping[binnumbers_i]
    
        #convert 1D binnumbers_i to equivalent ND indices
        x_idx = np.unravel_index(binnumbers_i, [len(b) + 1 for b in self.bins])
        x_idx = [x_idx[i] - 1 for i in range(self.N_c)]
    
        return self.rmean[x_idx].reshape([self.N, self.N]) #, self.rstd[x_idx].reshape([self.N, self.N])

    #Given c_i, randomly sample from sub bin averages, according to the jump pmfs
    def get_sub_mean_r_ip1(self, c_i):
        
        #find in which bins the c_i samples fall
        _, _, binnumbers_i = stats.binned_statistic_dd(c_i, np.zeros(self.N**2), bins=self.bins)
        
        #dynamically corrects binnumbers_i if outliers are found
        #self.check_outliers(binnumbers_i, c_i)

        #static correction for outliers, using precomputed mapping array
        binnumbers_i = self.mapping[binnumbers_i]
  
        #the selected jump pmfs for each spatial point
        pmfs = self.jump_pmfs[binnumbers_i]
        #N**2 random U(0, 1) samples to select the sub bin
        xi = np.random.rand(self.N**2)
        
        #draw a random sample from each pmf 
        sub_idx = [pmf_i.ppf(xi_i) for xi_i, pmf_i in zip(xi, pmfs)] #QUESTION: CAN THIS BE VECTORIZED? 
        sub_idx = np.array(sub_idx).astype('int')
        
        return self.r_sub_mean[binnumbers_i, sub_idx].reshape([self.N, self.N])

    #append the covariates supplied to the binning object during simulation
    #to self.covars
    #Note: use list to dynamically append, array is very slow
    def append_covar(self, c_i):
       
        for i in range(self.N_c):
            self.covar[i].append(c_i[:,i])
            
            #if max number of covariates is reached, remove first item
            if len(self.covar[i]) > self.max_lag:
                self.covar[i].pop(0)

    #return lagged covariates, assumes constant lag
    #Note: typecast to array if spatially varying lag is required
    def get_covar(self, lags):
        
        c_i = np.zeros([self.N**2, self.N_c])

        for i in range(self.N_c):
            
            if lags[i] <= self.max_lag:
                c_i[:, i] = self.covar[i][-lags[i]]
            else:
                print 'Warning, max lag exceeded'
                import sys; sys.exit()
            
        return c_i       

    #compute the probabilities of staying or jumping to the left/right of the current bin, 
    #when sampling randomly from each bin
    #ASSUMES AUTOCORRELATED SURROGATE WITH ONE COVARIATE
    def compute_surrogate_jump_probabilities(self, plot=False):

        if self.N_c != 1:
            print 'Only works for N_c = 1'
            return

        #store the jump probabilities of the surrogate per bin
        self.surrogate_jump_probs = np.zeros([self.N_bins+1, 3])

        #store the r sample means of each jump probability per bin
        self.r_sub_mean = np.zeros([self.N_bins+1, 3])

        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(111, xlabel=r'$\mathcal{C}_i$', ylabel=r'$\overline{r}_{i+1}$')

        for j in range(self.unique_binnumbers.size):

            J = self.unique_binnumbers[j]

            #indices of all r samples in the current bin
            idx = np.where(self.binnumber == J)[0]

            #r samples of the current bin
            r_idx = self.r_ip1[idx]

            if r_idx.size > 0:
                #indices of r_idx, IF USED AS COVARIATE
                _, _, bins_idx = stats.binned_statistic_dd(r_idx, np.zeros(r_idx.size), bins=self.bins)

                #number of samples in the current bin
                nsamples_bin = np.double(idx.size)

                print '++++++++++++++++++++'
                print 'Bin ', J

                #indices that are unchanged, larger or smaller than the current bin index
                same_idx = np.where(bins_idx== J)[0]
                left_idx = np.where(bins_idx < J)[0]
                right_idx = np.where(bins_idx > J)[0]

                #compute jump probabilities      
                self.surrogate_jump_probs[J, 0] = left_idx.size/nsamples_bin
                self.surrogate_jump_probs[J, 1] = same_idx.size/nsamples_bin
                self.surrogate_jump_probs[J, 2] = right_idx.size/nsamples_bin

                print 'Number of samples in bin =', nsamples_bin
                print 'Jump left probability = ', left_idx.size/nsamples_bin
                print 'Stay probability =', same_idx.size/nsamples_bin
                print 'Jump right probability =', right_idx.size/nsamples_bin

                #bin mean associated with each jump event
                self.r_sub_mean[J, 0] = np.mean(self.r_ip1[idx[left_idx]])
                self.r_sub_mean[J, 1] = np.mean(self.r_ip1[idx[same_idx]])
                self.r_sub_mean[J, 2] = np.mean(self.r_ip1[idx[right_idx]])

                #colorcode samples per bin based on jump behavior
                if plot == True:

                    #color code samples in current bin based on the bin that will be occupied in next iteration
                    ax.plot(self.c[idx[same_idx]], self.r_ip1[idx[same_idx]], '+', color='darkgray', alpha = 0.1)
                    ax.plot(self.c[idx[left_idx]], self.r_ip1[idx[left_idx]], '+', color='lightgray', alpha = 0.1)
                    ax.plot(self.c[idx[right_idx]], self.r_ip1[idx[right_idx]], '+', color='lightgray', alpha = 0.1)

                    #plot mean of the stay, jump left/right samples per bin
                    ax.plot(np.mean(self.c[idx[left_idx]]), self.r_sub_mean[J, 0], 'ko')
                    ax.plot(np.mean(self.c[idx[same_idx]]), self.r_sub_mean[J, 1], 'ko')
                    ax.plot(np.mean(self.c[idx[right_idx]]), self.r_sub_mean[J, 2], 'ko')

                    ax.vlines(self.bins, np.min(self.r_ip1), np.max(self.r_ip1))

                    plt.tight_layout()
                    #plt.savefig('jump.png', dpi=400)

    #Compute the probabilities on staying, jumping to the left or the right of the current bin
    #Based on the r data
    def compute_jump_probabilities(self):

        print 'Computing jump probabilities using', self.N_s, 'data fields'

        #reshape the binnumbers of the data 
        binnumbers_k = self.binnumber.reshape([self.N**2, self.N_s])

        #store the number/prob of jumps to the left, in place or to the right
        #size = N_bins+1 since bin indices start at 1
        self.jumps = np.zeros([self.N_bins+1, 3])
        self.jump_probs = np.zeros([self.N_bins+1, 3])
        self.jump_pmfs = [] 
        self.jump_pmfs.append([])

        for s in range(np.max(self.lags), self.N_s):
            #the binnumbers of the s-th data entry
            binnumbers_i = binnumbers_k[:, s-np.max(self.lags)]
            #the binnumbers of the (s+1)-th data entry
            binnumbers_ip1 = binnumbers_k[:, s]

            #no bin change indices
            no_jump_idx = np.where(binnumbers_i == binnumbers_ip1)[0]
            #jumped to the left indices
            jump_left_idx = np.where(binnumbers_i > binnumbers_ip1)[0]
            #jumped to the right indices
            jump_right_idx = np.where(binnumbers_i < binnumbers_ip1)[0]

            #count number of jumps (IN CASE OF A FIELD SAMPLE)
            #count_bins = np.arange(0.5, self.N_bins+1, 1)   #bins surrounding the binnumbers 1, 2, 3, ... (bins=[0.5, 1.5, 2.5, 3.5, ...])
            #jump_left_count, _, _ = stats.binned_statistic(binnumbers_i[jump_left_idx], np.zeros(jump_left_idx.size), statistic='count', bins=count_bins) 
            #stay_count, _, _ = stats.binned_statistic(binnumbers_i[no_jump_idx], np.zeros(no_jump_idx.size), statistic='count', bins=count_bins) 
            #jump_right_count, _, _ = stats.binned_statistic(binnumbers_i[jump_right_idx], np.zeros(jump_right_idx.size), statistic='count', bins=count_bins) 
            #
            #self.jumps[1:, 0] += jump_left_count 
            #self.jumps[1:, 1] += stay_count
            #self.jumps[1:, 2] += jump_right_count 

            jump_left_count = jump_left_idx.size
            stay_count = no_jump_idx.size
            jump_right_count = jump_right_idx.size

            self.jumps[binnumbers_i, 0] += jump_left_count
            self.jumps[binnumbers_i, 1] += stay_count
            self.jumps[binnumbers_i, 2] += jump_right_count

        #compute jump probabilities per bin
        for j in range(1, self.N_bins+1):
            self.jump_probs[j,:] = self.jumps[j,:]/np.sum(self.jumps[j,:])
            print 'Jump probabities bin ', j, '=', self.jump_probs[j, :]

            #create probability mass functions
            self.jump_pmfs.append(stats.rv_discrete(values=(range(3), self.jump_probs[j,:])))

        self.jump_pmfs = np.array(self.jump_pmfs)

    #plot the jump probability of the data and the surrogate per bin
    def plot_jump_pmfs(self):
       
        try:
            fig = plt.figure(figsize = [6.4, 5.8])
            x = np.arange(1, self.N_bins+1)

            #left probs
            ax = fig.add_subplot(311, ylabel=r'jump left prob.', xlabel=r'bin', ylim=[0, 1])
            ax.plot(x + 0.2, self.jump_probs[1:, 0], 'bo', ms=12, label=r'$\mathbb{P}\left(\bar{r}_{i+1}\in b_{j-k}\mid\bar{r}_i\in b_j\right)$')            
            ax.vlines(x + 0.2, 0, self.jump_probs[1:, 0], colors='b', lw=4)
            
            ax.plot(x - 0.2, self.surrogate_jump_probs[1:, 0], 'rs', ms=12,  label=r'$\mathbb{P}\left(\widetilde{r}_{i+1}\in b_{j-k}\mid\widetilde{r}_i\in b_j\right)$')            
            ax.vlines(x - 0.2, 0, self.surrogate_jump_probs[1:, 0], colors='r', lw=4)

            leg = plt.legend(loc=0)
            leg.draggable(True)
            ax.set_xticks(range(self.N_bins+1))

            #right probs
            ax = fig.add_subplot(312, ylabel=r'stay prob.', xlabel=r'bin', ylim=[0, 1])
            ax.plot(x + 0.2, self.jump_probs[1:, 1], 'bo', ms=12, label=r'$\mathbb{P}\left(\bar{r}_{i+1}\in b_{j}\mid\bar{r}_i\in b_j\right)$')            
            ax.vlines(x + 0.2, 0, self.jump_probs[1:, 1], colors='b', lw=4)
            
            ax.plot(x - 0.2, self.surrogate_jump_probs[1:, 1], 'rs', ms = 12, label=r'$\mathbb{P}\left(\widetilde{r}_{i+1}\in b_{j}\mid\widetilde{r}_i\in b_j\right)$')            
            ax.vlines(x - 0.2, 0, self.surrogate_jump_probs[1:, 1], colors='r', lw=4)
            
            leg = plt.legend(loc=0)
            leg.draggable(True)
            ax.set_xticks(range(self.N_bins+1))
            
            #right probs
            ax = fig.add_subplot(313, ylabel=r'jump right prob.', xlabel=r'bin', ylim=[0, 1])
            ax.plot(x + 0.2, self.jump_probs[1:, 2], 'bo', ms = 12, label=r'$\mathbb{P}\left(\bar{r}_{i+1}\in b_{j+k}\mid\bar{r}_i\in b_j\right)$')
            ax.vlines(x + 0.2, 0, self.jump_probs[1:, 2], colors='b', lw=4)
            
            ax.plot(x - 0.2, self.surrogate_jump_probs[1:, 2], 'rs', ms = 12, label=r'$\mathbb{P}\left(\widetilde{r}_{i+1}\in b_{j+k}\mid\widetilde{r}_i\in b_j\right)$')
            ax.vlines(x - 0.2, 0, self.surrogate_jump_probs[1:, 2], colors='r', lw=4)
            
            leg = plt.legend(loc=0)
            leg.draggable(True)
            ax.set_xticks(range(self.N_bins+1))

            plt.tight_layout()

        #if jump probabilities do not exist
        except AttributeError:
            plt.close()
            print 'Compute probabilities first'

    #c samples per bin stored as an array of arrays instead of 1d array
    def compute_c_samples_per_bin(self):

        c_per_bin = []
        
        for b in range(np.max(self.unique_binnumbers)+1):
            
            idx = np.where(self.binnumber == b)[0]
            c_per_bin.append(self.c[idx])
            
        self.c_per_bin = np.array(c_per_bin)
        
    #same as idx_per_bin only stored as a dict instead of 1d array
    def compute_binnumber_per_bin(self):

        idx_per_bin_dict = {}
        
        for b in self.unique_binnumbers:
            #r_ip1 indices of bin b
            idx_per_bin_dict[b] = np.where(self.binnumber == b)[0]
            
        self.idx_per_bin_dict = idx_per_bin_dict
        
    def plot_samples_per_bin(self, subsample=1):
        
        if self.N_c == 1:
        
            fig = plt.figure()
            ax = fig.add_subplot(111, xlabel=r'$\mathrm{conditioning\;variable}$', ylabel=r'$\mathrm{reference\;data}$')
            
            for b in self.unique_binnumbers:
                
                c_b = self.c[self.idx_per_bin_dict[b]]
                r_b = self.r_ip1[self.idx_per_bin_dict[b]]

                ax.plot(c_b[0:-1:subsample], r_b[0:-1:subsample], '+', color='lightgray', alpha=0.3)
                ax.plot(np.mean(c_b), np.mean(r_b), 'ko', markersize=6)
                
            ax.vlines(self.bins, np.min(self.r_ip1), np.max(self.r_ip1))
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.tight_layout()
            #plt.savefig('bins.png', dpi=300)

        plt.show()

    #plots the mean and std dev per bin, ASSUMES 2 COVARIATES
    def plot_bin_stats(self):
        
        if self.N_c != 2:
            print 'Only works for 2D c'
        else:
            
            fig = plt.figure(figsize=[8.0, 3.71])
            ax = fig.add_subplot(121)
            ax.set_title(r'$\mathbb{E}[r]$', fontsize=20)
            ax.set_xlabel(r'$c_1\;\mathrm{bins}$', fontsize=20)
            ax.set_ylabel(r'$c_2\;\mathrm{bins}$', fontsize=20)
            im = ax.imshow(self.rmean)
            plt.colorbar(im)
            #
            ax = fig.add_subplot(122)
            ax.set_title(r'$\sqrt{\mathbb{V}ar[r]}$', fontsize=20)
            ax.set_xlabel(r'$c_1\;\mathrm{bins}$', fontsize=20)
            ax.set_ylabel(r'$c_2\;\mathrm{bins}$', fontsize=20)
            im = ax.imshow(self.rstd)
            plt.colorbar(im)
            
            plt.tight_layout()
            plt.show()
            
    def print_bin_info(self):
        print '-------------------------------'
        print 'Total number of samples= ', self.r_ip1.size
        print 'Total number of bins = ', self.N_bins**self.N_c
        print 'Total number of non-empty bins = ', self.binnumbers_nonempty.size
        print 'Percentage filled = ', np.double(self.binnumbers_nonempty.size)/self.N_bins**self.N_c*100., ' %'
        print '-------------------------------'
        
    #compute the uniform bins of the conditional variables in c
    def get_bins(self, N_bins):
        
        bins = []
        
        for i in range(self.N_c):
            bins.append(np.linspace(np.min(self.c[:,i]), np.max(self.c[:,i]), N_bins+1))
    
        return bins
    
    def get_bins_same_nsamples(self, N_bins):
        
        if self.N_c != 1:
            print 'Only works for N_c = 1'
            return
        
        Ns = self.c.size
        
        delta = Ns/N_bins
        
        c_sorted = np.sort(self.c.reshape(Ns))
        
        bins = []
        bins.append(c_sorted[0])
        
        for i in range(N_bins-1):
            bins.append(c_sorted[(i+1)*delta])
            
        bins.append(c_sorted[-1])
            
        #plt.figure()
        #plt.plot(c_sorted, 'b')
        #plt.plot(np.zeros(N_bins+1), bins, 'ro')
        
        return [np.array(bins)]

def std_per_bin(x):
    return np.var(x)**0.5

import numpy as np
from scipy import stats
from itertools import chain, product
import matplotlib.pyplot as plt
import itertools
