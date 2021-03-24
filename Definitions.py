import numpy as np
import logging
import mne
import os
import matplotlib.pyplot as plt
from scipy.stats import median_absolute_deviation as mad
from scipy.stats import pearsonr
from copy import deepcopy
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.io.proj import _proj_equal, setup_proj, ProjMixin

###########################################################################################################
############################################## Preprocessing ##############################################
###########################################################################################################

#Preprocessing (Filtering, ICA, Bad channel detection):
def preprocessing (filename):
    "1. Makes raw data out of file, 2. Asks for plot (but is in #), 3. Filters the data (0.01, 45 Hz). 4. Uses definition `detect bad channels`, 5. Uses ICA" 

    #1. Make raw data:
    raw = mne.io.read_raw (filename, preload = True)

    # 2. Plotting raw data? 
    # if Plotting == 'Yes':
    #     raw.plot()


    #3. Filtering: ##--> tpye in 30 evtl. 
    lfreq, hfreq = (1, 30) # Before- (0.01, 45)
    raw.filter(lfreq,hfreq)
    raw.set_montage('standard_1020')


    #4. Bad channel detection anwenden:
    raw = detect_bad_chans (raw)
    raw.set_eeg_reference('average', projection = False)


    #5. ICA:
    #Filter (Highpass)
    raw_ica = deepcopy(raw)
    raw_ica.filter(1., None, n_jobs=-1)
    method = 'fastica'
    # Initialize & fit
    ica = mne.preprocessing.ICA(n_components=20, method=method, random_state= 97,   max_iter=200) 
    ica.fit(raw_ica)
    # Find artefactual components based on Fp1 activity
    ica.exclude = []
    eog_epochs = mne.preprocessing.create_eog_epochs(raw_ica, ch_name ='Fp1')
    # eog_average = eog_epochs.average()
    eog_inds, scores = ica.find_bads_eog(eog_epochs, ch_name ='Fp1')  # find via correlation
    ica.exclude = eog_inds
    print ('Eog indices: ', eog_inds)
    # Remove said artefactual components
    ica.apply(raw)
    
    #Logging info
    log_msg = f'Remove component {ica.exclude}'
    logging.info(log_msg)

    # Plot 
    # raw.load_data()
    # ica.plot_sources(raw, show_scrollbars=False)
    # ##Visualize the scalp field distribution of each component
    # fig= ica.plot_components()
    # ##fig.savefig('plot.png') ##########################################################################

    # ##We can also plot some diagnostics of each IC using:
    # ica.plot_properties(raw, picks=[0])

    return(raw)

#Get events- is used in Get_Epochs below! 
def get_events(annotations):
    '''Get events- is used in Get_Epochs'''
    a = annotations[0]
    evnts = np.zeros((a.shape[0], a.shape[1]+1))
    evnts[:,0:3] = a
    
    for i in range(evnts.shape[0]):
        if evnts[i,2] == 13:
            j = i+1
            while j<evnts.shape[0]:
                if evnts[j,2] == 12:
                    evnts[i,3] = np.round((evnts[j,0] - evnts[i,0])/50)*50
                    if evnts[i,3] != 50:
                        evnts[i,2] = evnts[i,3]
                    break
                j+=1
    return np.ndarray.astype(evnts[:,0:3],'int32')

# Get the Epochs:
def Get_Epochs (raw):
    "1. makes Annotations, 2. Gets events out of annotations, 3. Assigns same Trigger, 4. Gets Epochs: Ambiguous, Unambiguous, Low Noise, High Noise--> TAKES: RAW/ RETURNS: EPOCHS"
    #1. Annotations: 
    annotations = mne.events_from_annotations (raw)
    #triggermat = annotations [0] #############################################


    #2. Get events with def above:
    events = get_events (annotations)
    # print ('EVENTS SHAPE', events.shape)
    # print ('Lenght', len(events[:,2]))

    #Assign events to the same trigger - Unambigouos= 500, Low Noise= 450, High Noise= 650
    for count, value in enumerate (events[:,2]):
        if value == 400:
            events[count,2] = 500
        if value == 550:
            events[count,2] = 450
        if value == 700:
            events[count,2] = 650

    #3. Get epochs with events:
    event_id = {'Ambiguous': 600,
                'Unambiguous': 500,
                'Low Noise' : 450, 
                'High Noise': 650,
                 }   
           
    tmin = -0.2 
    tmax = 0.8 
    baseline = (-0.06, 0.04) # BASELINE--> Can be set to "None"
    reject = dict(eeg = 200.0*1e-6)

    epochs = mne.Epochs(raw, events = events, event_id = event_id, baseline = baseline, reject = reject, tmin = tmin, tmax = tmax, event_repeated = "merge", on_missing = 'ignore',        preload = True)
    epochs.set_montage('standard_1020')
    epochs.set_eeg_reference('average', projection=True)

    return (epochs)

#Functions that are used during preprocessing: 
#Bad channel detection:
def detect_bad_chans(raw, chan_list_ordererd=None, crit=0.2, radius=0.075, zthresh=100):
    ''' Detect bad channels and return list of strings 
    radius...in meters within which neighboring electrodes are defined
    crit... neighbor correlation criterion. If maximum correlation with any neighbor is below this crit the channels is marked as bad
    '''

    if chan_list_ordererd is None:
        chan_list_ordererd = get_chan_pos_list(raw)

    # Handle exceptions
    assert len(chan_list_ordererd) == len(raw.ch_names), 'Chan list has different number of channels than there are channels in raw'

    # Correlation Criterion "CC"
    bad_chans_cc = bad_chan_cc(raw, chan_list_ordererd, crit, radius)
    print(f'Bad because of correlation: {bad_chans_cc}')
    # Deviation Criterion "DC"
    bad_chans_dc = bad_chan_dc(raw, zthresh)
    print(f'Bad because of deviation: {bad_chans_dc}')
    # Summarize
    bad_chans = np.unique(np.concatenate((bad_chans_cc, bad_chans_dc)))
    
    # Bad Channel info
    bads_info = []
    for i in bad_chans:
        if i in bad_chans_cc and i in bad_chans_dc:
            bads_info.append('Corr & Deviation')
        elif i in bad_chans_cc and not i in bad_chans_dc:
            bads_info.append('Corr')
        else:
            bads_info.append('Deviation')

    raw.info['bads'] = bad_chans
    raw.info['bads_info'] = bads_info
    return raw

#Detected Outlier-  Z-Axis 
def bad_chan_dc(raw, zthresh):
    ''' Used in detect_bad_chans '''
    bad_chans = []
    
    total_median = np.median(raw._data)
    total_mad = mad(raw._data)

    for i in range(len(raw.ch_names)):
        data = raw._data[i, :]
        # Robust Z-Scaling
        zdata = (data - total_median) / total_mad
        # Median Z-score
        zscore = np.median(np.abs(zdata))
        # If high amplitudes compared to rest:
        if zscore > zthresh:
            bad_chans.append(raw.ch_names[i])
        print(f'ch {raw.ch_names[i]} has z of {zscore} (zthresh={zthresh})')
    return bad_chans


# Checks every channel and calculates correlation to the neighbour correlation < 0.2
def bad_chan_cc(raw, chan_list_ordererd, crit, radius):
    ''' This function checks the correlation criterion (cc) as a base for rejecting bad channels: 
        If a channel is not sufficiently correlated with its close neighbors, it is probably bad --> Used in detect_bad_chans'''
    bad_chans = []
    # Loop through each channel
    for i in range(len(raw.ch_names)):
        # Get position of current channel
        pos = chan_list_ordererd[i][1]
        # Find neighbors
        distlist = np.zeros((len(raw.ch_names)-1))
        pos_list = [ls[1] for j, ls in enumerate(chan_list_ordererd) if j != i]
        pos_list = np.array(pos_list)

        idx_list = [j for j in range(len(chan_list_ordererd)) if j != i]

        for j in range(pos_list.shape[0]):
            distlist[j] = euclidean_distance(pos_list[j, :], pos)
        neighbor_indices = np.argwhere(distlist<=radius)
        neighbor_indices = [k[0] for k in neighbor_indices]
        neighbor_indices = [idx_list[k] for k in neighbor_indices]
        # ...
        # Perform Correlations
        corrs = []
        for k, index in enumerate(neighbor_indices):
            corrs.append(pearsonr(raw._data[i, :], raw._data[index, :])[0])
        
        if np.max(np.abs(corrs)) < crit:
            print(f'{raw.ch_names[i]} is bad!')
            bad_chans.append(raw.ch_names[i])
        else:
            print(f'{raw.ch_names[i]} is fine! (corr={np.max(np.abs(corrs))}) {len(neighbor_indices)} neighbors')

    return bad_chans

#Ordered channel list: 
def get_chan_pos_list(raw, montage='standard_1020'):
    ''' Used in detect_bad_chans '''
    if type(montage) == str:
        montage = mne.channels.make_standard_montage(montage)
    
    chan_list = [[i] for i in montage.ch_names]
    [chan_list[i].append(montage.dig[i]['r']) for i in range(len(montage.ch_names))]
    ordered_chan_list = []
    for ch in raw.ch_names:
        for row in chan_list:
            if row[0] == ch:
                break
        ordered_chan_list.append(row)
    return ordered_chan_list

#euclidean_distance
def euclidean_distance(x, y):
    ''' Calculate euclidean distance between two vectors x and y --> used in bad_chan_cc'''
    return np.sqrt(np.sum((x-y)**2))

################################################################################################################################
######################################## Different functions to use for ITV calculation ########################################
################################################################################################################################

# def  Intertrial_variability (epochs, ch_name) :
#     ch_idx = epochs.ch_names.index(ch_name)
#     times_epoch = epochs.times

#     data_varia = epochs.get_data()[:, ch_idx, :]
#     print (data_varia.shape)
#     itv_varia = np.std(data_varia, axis = 0)
#     print (itv_varia.shape)

#     plt.figure()
#     plt.plot(times_epoch, itv_varia, label= 'itv')
#     plt.legend()
#     plt.show


# def itv_topo_map (epochs):
#     data = epochs.get_data()
#     itv = np.std(data, axis=0)
#     itv.shape
#     t = 0.6
#     pnt = np.where (epochs.times==t)[0][0]
#     plt.figure()
#     mne.viz.plot_topomap(itv[:, pnt], epochs.info)

# def channel_plot (epochs, channel_name):
#     Cz_data = epochs.get_data()
#     index = epochs.ch_names.index(channel_name) 
#     plt.figure()
#     plt.plot (Cz_data[4, index, :])
#     plt.ylim([-100e-6, 100e-6])

# def ERP_Plot (epochs):
#     evoked = epochs.average()
#     times = (0.100, 0.120, 0.160, 0.180, 0.200)
#     evoked.plot()
#     evoked.plot_topo()
#     evoked.plot_topomap(times)
#     return (evoked)

# def Unamb_Amb_itv (List_Conc_per_person, LOOPLEN= 'ASDFG'):
#     "Def. to search Unamb und Amb and to calculate ITV of them --> TAKES: List of concatenated epochs from people AND Len of the Loop (len (List_of_People))/ RETURNS: List of itv from Unamboguous (Unamb_List) and Ambiguous (Amb_List)"
#     Unamb_List= []
#     Amb_List= []
#     for indx in range ((len(LOOPLEN))):                                                  
#         data_Unamb= List_Conc_per_person[indx]['Unambiguous'].get_data()
#         itv_Unamb = np.std(data_Unamb, axis = 0)
#         Unamb_List.append(itv_Unamb)

#         data_Amb= List_Conc_per_person[indx]['Ambiguous'].get_data()
#         itv_Amb = np.std(data_Amb, axis = 0)
#         Amb_List.append(itv_Amb)
#     return (Unamb_List, Amb_List)

####################################################################################################################
######################################### Definition to loop through files #########################################
####################################################################################################################

def list_files(dir):
    "Looping through files and Subfiles--> --> TAKES: direction/ RETURNS: List of subfiles "
    r = []
    for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith(".vhdr") and not "_C" in name and not name.startswith('LR') and not name.startswith('SR') and not name.startswith('C') and not '._' in name:
                    #Dafür wurden die Daten in dem Ordner umbenannt- alle CR, CL und LR sowie SR zu CLR und CSR wurde die Reihenfolge getauscht--> Wieder rückgängig gemacht um alle Datensätze verarbeiten zu können
                    #C = Checker board- L= Large, S= Small, R= Rare
                    r.append(os.path.join(root, name))
    return r

#####################################################################################################################################
################################################ Different ways to calculate evokeds ################################################
#####################################################################################################################################

def make_evoked (List_Conc_per_person, LOOPLEN= 'ASDFG'):
    "Def. to make List of concatenated epochs in evoked- used for better and easier plots--> TAKES: List of concatenated epochs from people AND Len of the Loop (len (List_of_People))/ RETURNS: List of itv from the single people "
    New_evoked_itv = []
#https://mne.tools/dev/auto_tutorials/simulation/plot_creating_data_structures.html#id4
    for indx in range ((len(LOOPLEN))):                                                  
        evoked_array = mne.EvokedArray(np.std(List_Conc_per_person[indx].get_data(), axis=0), List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        New_evoked_itv.append(evoked_array)
    return (New_evoked_itv)


def itv_evoked (List_Conc_per_person, LOOPLEN= 'ASDFG'):
    "Def. to make List of evoked ITVs from only ambiguous/ unambgiguous/ Low noise/ High noise--> TAKES: List of concatenated epochs from people AND Len of the Loop (len (List_of_People))/ RETURNS: Evoked Lists: Ambiguous, Unambiguous, Low Noise, High Noise  "
    Evoked_itv_Ambiguous = []
    Evoked_itv_Unambiguous = []
    Evoked_itv_LowNoise = []
    Evoked_itv_HighNoise = []

    for indx in range ((len(LOOPLEN))):  
        data_Unamb = List_Conc_per_person[indx]['Unambiguous'].get_data()
        itv_Unamb = np.std(data_Unamb, axis = 0)                                                
        evoked_array_Unamb = mne.EvokedArray(itv_Unamb, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_Unambiguous.append(evoked_array_Unamb)

        data_Amb = List_Conc_per_person[indx]['Ambiguous'].get_data()
        itv_Amb = np.std(data_Amb, axis = 0)
        evoked_array_Amb = mne.EvokedArray(itv_Amb, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_Ambiguous.append(evoked_array_Amb)

        
        data_LN = List_Conc_per_person[indx]['Low Noise'].get_data()
        itv_LN = np.std(data_LN, axis = 0)
        evoked_array_LN = mne.EvokedArray(itv_LN, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_LowNoise.append(evoked_array_LN)

        data_HN = List_Conc_per_person[indx]['High Noise'].get_data()
        itv_HN = np.std(data_HN, axis = 0)
        evoked_array_HN = mne.EvokedArray(itv_HN, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_HighNoise.append(evoked_array_HN)

    return (Evoked_itv_Ambiguous, Evoked_itv_Unambiguous, Evoked_itv_LowNoise, Evoked_itv_HighNoise) 

def CV_evoked (List_Conc_per_person, LOOPLEN= 'ASDFG'):
    """Def. to make List of Coefficient of variation from only ambiguous/ unambgiguous/ Low noise/ high noise--> TAKES: List of concatenated epochs from people AND Len of the Loop (len (List_of_People))/ RETURNS: Evoked Lists: Ambiguous, Unambiguous, Low Noise, High Noise"""
    Evoked_itv_Ambiguous = []
    Evoked_itv_Unambiguous = []
    Evoked_itv_LowNoise = []
    Evoked_itv_HighNoise = []

    for indx in range ((len(LOOPLEN))):  
        data_Unamb = List_Conc_per_person[indx]['Unambiguous'].get_data()
        rms_Unamb = np.sqrt(np.mean(np.mean(data_Unamb, axis = 0)**2))
        adder_Unamb = np.abs(np.min(data_Unamb)*2)
        itv_Unamb = np.divide(np.std(data_Unamb, axis = 0), rms_Unamb)                                               
        evoked_array_Unamb = mne.EvokedArray(itv_Unamb, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_Unambiguous.append(evoked_array_Unamb)

        data_Amb = List_Conc_per_person[indx]['Ambiguous'].get_data()
        rms_Amb = np.sqrt(np.mean(np.mean(data_Amb, axis = 0)**2))
        adder_Amb = np.abs(np.min(data_Amb)*2)
        itv_Amb = np.divide(np.std(data_Amb, axis = 0), rms_Amb)
        evoked_array_Amb = mne.EvokedArray(itv_Amb, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_Ambiguous.append(evoked_array_Amb)

        
        data_LN = List_Conc_per_person[indx]['Low Noise'].get_data()
        rms_LN = np.sqrt(np.mean(np.mean(data_LN, axis = 0)**2))
        adder_LN = np.abs(np.min(data_LN)*2)
        itv_LN = np.divide(np.std(data_LN, axis = 0), rms_LN)
        evoked_array_LN = mne.EvokedArray(itv_LN, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_LowNoise.append(evoked_array_LN)

        data_HN = List_Conc_per_person[indx]['High Noise'].get_data()
        rms_HN = np.sqrt(np.mean(np.mean(data_HN, axis = 0)**2))
        adder_HN = np.abs(np.min(data_HN)*2)
        itv_HN = np.divide(np.std(data_HN, axis = 0), rms_HN)
        evoked_array_HN = mne.EvokedArray(itv_HN, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_itv_HighNoise.append(evoked_array_HN)

    return (Evoked_itv_Ambiguous, Evoked_itv_Unambiguous, Evoked_itv_LowNoise, Evoked_itv_HighNoise) 

def make_evoked_mean (List_Conc_per_person, LOOPLEN= 'ASDFG'):
    "Def. to make List of evoked from only ambiguous/ unambgiguous/ Low noise/ High noise--> TAKES: List of concatenated epochs from people AND Len of the Loop (len (List_of_People))/ RETURNS: Evoked Lists: Ambiguous, Unambiguous, Low Noise, High Noise  "
    Evoked_Ambiguous = []
    Evoked_Unambiguous = []
    Evoked_LowNoise = []
    Evoked_HighNoise = []

    for indx in range ((len(LOOPLEN))):  
        data_Unamb = List_Conc_per_person[indx]['Unambiguous'].get_data()      
        mean_Unamb = np.mean(data_Unamb, axis = 0)                                                                           
        evoked_array_Unamb = mne.EvokedArray(mean_Unamb, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_Unambiguous.append(evoked_array_Unamb)

        data_Amb = List_Conc_per_person[indx]['Ambiguous'].get_data()
        mean_Amb = np.mean(data_Amb, axis = 0)
        evoked_array_Amb = mne.EvokedArray(mean_Amb, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_Ambiguous.append(evoked_array_Amb)

        
        data_LN = List_Conc_per_person[indx]['Low Noise'].get_data()
        mean_LN = np.mean(data_LN, axis = 0)
        evoked_array_LN = mne.EvokedArray(mean_LN, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_LowNoise.append(evoked_array_LN)

        data_HN = List_Conc_per_person[indx]['High Noise'].get_data()
        mean_HN = np.mean(data_HN, axis = 0)
        evoked_array_HN = mne.EvokedArray(mean_HN, List_Conc_per_person[indx].info, tmin=-0.2, comment='simulated')
        Evoked_HighNoise.append(evoked_array_HN)

    return (Evoked_Ambiguous, Evoked_Unambiguous, Evoked_LowNoise, Evoked_HighNoise) 

#######################################################################################################################################
############################################### Definitions for time frequency analysis ###############################################
#######################################################################################################################################

def get_itc(epoch):
    ''' Definition to investigate the time Frequency analysis'''
    freqs = np.logspace(*np.log10([4, 40]), num=50)
    n_cycles = freqs / 3. 
    kwargs = dict(n_jobs=-1,
                return_itc=True,
                average=True)
    
    power, itc = mne.time_frequency.tfr_morlet(epoch, freqs, n_cycles, **kwargs)
    
    return itc.data

def get_itpv(epoch):
    ''' Definition to investigate the time Frequency analysis'''
    freqs = np.logspace(*np.log10([4, 40]), num=50)
    n_cycles = freqs / 3. 
    kwargs = dict(n_jobs=-1,
                return_itc=False,
                average=False)
    
    power = mne.time_frequency.tfr_morlet(epoch, freqs, n_cycles, **kwargs)
    # power.apply_baseline(baseline=(-0.1, 0), mode='logratio')

    itpv = np.std(power.data, axis=0)
    
    return itpv

##############################################################################################################################
#################################### Definition to concatenate all epochs into a big list ####################################
##############################################################################################################################

def concatenate_epochs(epochs_list, with_data=True, add_offset=True):
    """Auxiliary function for concatenating epochs. Ruthlessly stolen from python package MNE. """
    if not isinstance(epochs_list, (list, tuple)):
        raise TypeError('epochs_list must be a list or tuple, got %s'
                        % (type(epochs_list),))
    # for ei, epochs in enumerate(epochs_list):
        # if not isinstance(epochs, epochs_var):
        #     raise TypeError('epochs_list[%d] must be an instance of epochs_var, '
        #                     'got %s' % (ei, type(epochs)))

    out = epochs_list[0]
    data = [out.get_data()] if with_data else None
    # print ('THE OUT TYPE', type(out))
    # print ('THE DATA TYPE', type(data))  
    events = [out.events]
    # print ('THE EVENTS', events) 
    # print ('THE EVENTS TYPE', type(events)) 


    metadata = [out.metadata]
    baseline, tmin, tmax = out.baseline, out.tmin, out.tmax
    info = deepcopy(out.info)
    verbose = out.verbose
    drop_log = deepcopy(list(out.drop_log))
    event_id = deepcopy(out.event_id)
    selection = out.selection
    # offset is the last epoch + tmax + 10 second
    events_offset = (np.max(out.events[:, 0]) +
                     int((10 + tmax) * epochs_list[0].info['sfreq']))
    for ii, epochs in enumerate(epochs_list[1:]):
        _compare_epochs_infos(epochs.info, info, ii)
        if not np.allclose(epochs.times, epochs_list[0].times):
            raise ValueError('Epochs must have same times')

        if epochs.baseline != baseline:
            raise ValueError('Baseline must be same for all epochs')

        # compare event_id
        common_keys = list(set(event_id).intersection(set(epochs.event_id)))
        for key in common_keys:
            if not event_id[key] == epochs.event_id[key]:
                msg = ('event_id values must be the same for identical keys '
                       'for all concatenated epochs. Key "{}" maps to {} in '
                       'some epochs and to {} in others.')
                raise ValueError(msg.format(key, event_id[key],
                                            epochs.event_id[key]))

        if with_data:
            data.append(epochs.get_data())
        evs = epochs.events.copy()
        # add offset
        if add_offset:
            evs[:, 0] += events_offset
        # Update offset for the next iteration.
        # offset is the last epoch + tmax + 10 second
        events_offset += (np.max(epochs.events[:, 0]) +
                          int((10 + tmax) * epochs.info['sfreq']))
        events.append(evs)
        selection = np.concatenate((selection, epochs.selection))
        drop_log.extend(list(epochs.drop_log)) 
        event_id.update(epochs.event_id)
        metadata.append(epochs.metadata)
    events = np.concatenate(events, axis=0)

    # Create metadata object (or make it None)
    n_have = sum(this_meta is not None for this_meta in metadata)
    if n_have == 0:
        metadata = None
    elif n_have != len(metadata):
        raise ValueError('%d of %d epochs instances have metadata, either '
                         'all or none must have metadata'
                         % (n_have, len(metadata)))
    else:
        pd = True  # _check_pandas_installed(strict=False)
        if pd is not False:
            metadata = pd.concat(metadata)
        else:  # dict of dicts
            metadata = sum(metadata, list())
    if with_data:
        data = np.concatenate(data, axis=0)
    for epoch in epochs_list:
        epoch.drop_bad()
    epochs = epochs_list[0]
    epochs.info = info
    epochs._data = data
    epochs.events = events
    epochs.event_id = event_id
    # epochs.tmin = tmin
    # epochs.tmax = tmax
    # epochs.metadata = metadata
    # epochs.baseline = baseline
    epochs.selection = selection
    epochs.drop_log = drop_log
    # return (info, data, events, event_id, tmin, tmax, metadata, baseline,
    #         selection, drop_log, verbose)
    return epochs

def _compare_epochs_infos(info1, info2, ind):
    """Compare infos."""
    info1._check_consistency()
    info2._check_consistency()
    if info1['nchan'] != info2['nchan']:
        raise ValueError('epochs[%d][\'info\'][\'nchan\'] must match' % ind)
    # if info1['bads'] != info2['bads']:
    #     raise ValueError('epochs[%d][\'info\'][\'bads\'] must match' % ind)
    if info1['sfreq'] != info2['sfreq']:
        raise ValueError('epochs[%d][\'info\'][\'sfreq\'] must match' % ind)
    if set(info1['ch_names']) != set(info2['ch_names']):
        raise ValueError('epochs[%d][\'info\'][\'ch_names\'] must match' % ind)
    if len(info2['projs']) != len(info1['projs']):
        raise ValueError('SSP projectors in epochs files must be the same')
    # if any(not _proj_equal(p1, p2) for p1, p2 in
    #        zip(info2['projs'], info1['projs'])):
    #     raise ValueError('SSP projectors in epochs files must be the same')
    if (info1['dev_head_t'] is None) != (info2['dev_head_t'] is None) or \
            (info1['dev_head_t'] is not None and not
             np.allclose(info1['dev_head_t']['trans'],
                         info2['dev_head_t']['trans'], rtol=1e-6)):
        raise ValueError('epochs[%d][\'info\'][\'dev_head_t\'] must match. '
                         'The epochs probably come from different runs, and '
                         'are therefore associated with different head '
                         'positions. Manually change info[\'dev_head_t\'] to '
                         'avoid this message but beware that this means the '
                         'MEG sensors will not be properly spatially aligned. '
                         'See mne.preprocessing.maxwell_filter to realign the '
                         'runs to a common head position.' % ind)

##########################################################################################################
#################################### TOPO PLOT FROM LUKAS AND MAREIKE ####################################
##########################################################################################################

from scipy.stats import sem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne 

def plot_topomap(epoch_list, error='sem', condition=None, channel_label_offsets=[0.15, 0.6],
    sem_alpha=0.5, window_margins=None, title='', legend=None, colors='default',
    plot_vert=False, linewidth=1, figsize=(9, 7), font_scale=0.8, font='georgia',
    convert_units=True, ylim_tol=0.2, plot_gap=0.04, time_range=None):
    '''
    Parameters:
    -----------
    epochs_list : list, a list of mne.Epochs objects
    error : str, defines the type of error shadings. Can be either
        'std', 'sem' or None.
    condition : str, defines the condition to be extracted from epochs
    channel_label_offsets : list, [x_offset, y_offset] defines the offsets of the channel 
        label from the leftmost position and the top of the axis
    sem_alpha  : float, the transparency value (alpha) for error shadings
    window_margins : list, [left_gap, right_gap, bottom_gap, top_gap] the 
        minimum allowed gap between the plots and the window margins
    title : str, title of the plot
    legend : list, legend names for the plot
    colors : str/list, can be either 'default' for the default color palette or 
        a custom list of matplotlib-readable color strings
    plot_vert : bool, plot vertical axes
    linewidth : float, line width of the ERPs
    figsize : tuple, size of the whole topoplot figure
    font_scale : float, determines the font size
    font : str, default is 'georgia'
    ylim_tol : float, the higher the wider the y-axis limits
    plot_gap : float, defines the minimum gap (i.e. spacing) between channel plots
    '''

    # Handle input list
    if type(epoch_list[0]) != list:
        # single list of epochs, i.e. single group
        epoch_list = [epoch_list]

    # Extract channel positions
    epoch = epoch_list[0][0]
    montage = epoch.get_montage()
    digs = montage.dig[3:]
    pos = np.stack([dig['r'][:2] for dig in digs], axis=0)
    times = epoch.times

    if time_range is None:
        pnt_range = np.arange(len(times))
    else:
        pnt_range = [np.argmin(np.abs(times-time_range[0])), np.argmin(np.abs(times-time_range[1]))]
        pnt_range = np.arange(*pnt_range)
    


    # Calculate ERP and measurement errors
    ERP, SEM = group_data(epoch_list, condition, error)
 
    if convert_units:
        ERP *= 1e6
        SEM *= 1e6

    # Extract time range of interest
    ERP = ERP[..., pnt_range]
    SEM = SEM[..., pnt_range]
    times = times[..., pnt_range]

    # Handle Data Shape
    if len(ERP.shape) < 3:
        # Single Group needs empty first dim:
        ERP = np.expand_dims(ERP, axis=0)
    if SEM is not None:
        if len(SEM.shape) < 3:
            # Single Group needs empty first dim:
            SEM = np.expand_dims(SEM, axis=0)

    #  Plot properties

    ## Sizes and gaps

    ### Handle sizes and distances
    if window_margins is None:
        window_margins = [0.1, 0.15, 0.1, 0.2]

    ### Define sizes and distances
    left_gap, right_gap, bottom_gap, top_gap = window_margins
    pos_binned = tidy_channel_positions(pos)
    width, height = get_plot_size(pos_binned, gap=plot_gap)
    ylim = [np.min(ERP)-abs(ylim_tol*np.min(ERP)) , np.max(ERP)+abs(ylim_tol*np.max(ERP))]

    ## Colors
    if colors == 'default':
        colors = sns.color_palette()

    ## Legend
    if type(legend) == str:
        legend = [legend]
    elif type(legend) == list:
        if len(legend) != ERP.shape[0]:
            print(f'Legend has {len(legend)} entries, although ERP indicates {ERP.shape[0]} groups')
            legend = None

    # Create distance between plots and the frame (size=(1,1))
    pos_binned[:, 0] = norm_to_range(pos_binned[:, 0], 0+left_gap, 1-right_gap)
    pos_binned[:, 1] = norm_to_range(pos_binned[:, 1], 0+bottom_gap, 1-top_gap)

    fig = plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale, style='ticks', context='notebook', font=font)
    plt.suptitle(title)
    plt.axis('off')

    if legend is not None:
        patches = list()
        for leg, color in zip(legend, colors):
            patches.append( mpatches.Patch(color=color, label=leg) )
            
        plt.legend(handles=patches, loc='lower right', bbox_to_anchor=(1.1, 0))
    print(f'pos_binned: {pos_binned.shape}')
    
    for ch, pos_, ch_name in zip(np.flip(np.arange(pos_binned.shape[0])), reversed(pos_binned), reversed(epoch.ch_names)):
    # for ch, (pos_, ch_name) in enumerate(zip(pos_binned, epoch.ch_names)):
        ax = plt.axes(np.append(pos_, [width, height]))
        plt.axis('off')
        # Horizontal Line
        plt.plot([times.min(), times.max()], [0, 0], color='black')
        # Vertical Line
        if plot_vert:
            plt.plot([0, 0], ylim, color='black')
        plt.ylim(ylim)
        ax.text(times.min()+channel_label_offsets[0]*(times.max()-times.min()),
            ylim[1]*channel_label_offsets[1], 
            ch_name,
            horizontalalignment='center')
        # Loop through groups
        for group in range(ERP.shape[0]):
            if SEM is not None:
                ax.fill_between(times, ERP[group, ch]-SEM[group, ch], ERP[group, ch]+SEM[group, ch], alpha=sem_alpha, color=colors[group])
            ax.plot(times, ERP[group, ch], color=colors[group], linewidth=linewidth)
            

        # Plot axes info on botto left plot:
        if pos_[1] == np.min(pos_binned[:, 1]) and pos_[0] == np.min(pos_binned[np.where(pos_binned[:, 1]==pos_[1])[0], 0]):
            plt.axis('on')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.ylabel('Amplitude [\u03BCV]')
            plt.xlabel('Time [s]')
    return fig


def discretize(arr, n_bins):
    if type(arr) == list:
        arr = np.array(arr)
    arr_bins = np.zeros((len(arr)))
    bins = np.arange(n_bins)
    boarders = np.linspace(arr.min(), arr.max(), n_bins)
    for bin in range(n_bins-1):
        indices_in_current_bin = (arr >= boarders[bin]) & (arr <= boarders[bin+1])
        arr_bins[indices_in_current_bin] = boarders[bin]
    return arr_bins

def tidy_channel_positions(pos, n_bins_vert=7):
    ''' Tidy up the channel position with a row-structure
    Parameters:
    -----------
    pos : numpy.ndarray, n_channelsx2 list of x and y coordinates of 
        channel positions
    n_bins_vert : int, number of rows to arrange electrodes to.
    Return:
    -------
    pos_binned : numpy.ndarray, the tidied up channel positions
    '''
    bins_vert = np.linspace(pos[:, 1].min(), pos[:, 1].max(), num=n_bins_vert)
    bins_y = discretize(pos[:, 1], n_bins_vert)

    ## Row binning
    bins_x = np.zeros((len(bins_y))) 
    for vertpos in np.unique(bins_y):
        # Get indices of electrodes in current row
        row_indices = np.where(bins_y==vertpos)[0]
        # Know how to sort and unsort them
        sort_ascend = np.argsort(pos[row_indices, 0])
        unsort_ascend = np.argsort(sort_ascend)
        # Find x positions that are equally spaced:
        new_row_pos = np.linspace(pos[row_indices, 0].min(), pos[row_indices, 0].max(), num=len(pos[row_indices, 0]))
        # Put the ascending values in the previous order
        new_row_pos = new_row_pos[unsort_ascend]
        bins_x[row_indices] = new_row_pos

    pos_binned = np.stack([bins_x, bins_y,], axis=1)
    # Normalize
    pos_binned[:, 0] -= pos_binned[:, 0].min()
    pos_binned[:, 0] /= pos_binned[:, 0].max()
    pos_binned[:, 1] -= pos_binned[:, 1].min()
    pos_binned[:, 1] /= pos_binned[:, 1].max()
    
    return pos_binned

def norm_to_range(arr, lo, hi):
    return lo + ((arr - arr.min()) * (hi - lo) / (arr.max() - arr.min()))

def get_plot_size(pos, gap=0.05):
    '''Get maximum window size for plots
    Parameters:
    -----------
    pos : numpy.ndarray, n_channelsx2 list of x and y coordinates of 
        channel positions
    gap : float, defines the minimal gap between axes
    Return:
    -------
    width : float, maximal allowed width of axes
    height : float, maximal allowed height of axes
    '''

    # Width
    ## Find the closest two plots can get horizontally
    row_bins = np.unique(pos[:, 1])
    n_plots = []
    min_dist = 9999
    for row_bin in row_bins:
        # for col in 
        n_plots.append( len(np.where(pos[:, 1]==row_bin)[0]) )
        min_row_dists = np.min(np.diff(np.sort(pos[np.where(pos[:, 1]==row_bin)[0], 0])))
        min_dist = np.min([min_row_dists, min_dist])

    width = min_dist - gap
    height = np.min(np.diff(np.sort(row_bins))) - gap
    return width, height

def group_data(all_data,cond,error):
    '''Get mean and error of group
    Parameters:
    ----------------------
    all_data: list, nested list of epochs or evoked objects for every participant in group
    cond: str, condition ID
    error: str, type of error 'sem' or 'std'
    Output:
    -----------------
    mean: numpy.ndarray, ERP for each group and/or condition. group and/or condition x elecs x datapoints
    error: numpy.ndarray, same shape as mean array.'''
    
    if isinstance(all_data[0][0], mne.epochs.Epochs):
        # call on functions to get epochs and epoch averages
        group = get_epoch_for_cond(all_data, cond)
        group_avg = [evoked_list(group) for group in group]
    
    if isinstance(all_data[0][0], mne.evoked.EvokedArray):
        #get average data from evoked objects
        data_array = []
        for group in all_data:
            group_list = []
            for participant in group:
                participant_data=participant.data
                group_list.append(participant_data)
            data_array.append(group_list)
        group_avg = data_array
    
    #get group averages and errors from averaged participant data
    group_avg_error = [group_mean_error(group_avg, error) for group_avg in group_avg]
    
    #reshape so all group data/condition data is in single 3d array
    mean = np.stack([group[0] for group in group_avg_error])
    
    if error is not None:
        error = np.stack([group[1] for group in group_avg_error])

    return mean, error

# def group_data(all_data_epoch,cond,error):
#     '''Get mean and error of group
#     Parameters:
#     ----------------------
#     all_data_epoch: list, nested list of epochs for every participant in group
#     cond: str, condition ID
#     error: str, type of error 'sem' or 'std'
#     Output:
#     -----------------
#     mean: numpy.ndarray, ERP for each group and/or condition. group and/or condition x elecs x datapoints
#     error: numpy.ndarray, same shape as mean array.'''
    
#     # call on functions to get epochs, average epochs, get errors
#     group = get_epoch_for_cond(all_data_epoch, cond)
    
#     group_avg = [evoked_list(group) for group in group]
    
#     group_avg_error = [group_mean_error(group_avg, error) for group_avg in group_avg]
    
#     #reshape so all group data/condition data is in single 3d array
#     mean = np.stack([group[0] for group in group_avg_error])
    
#     if error is not None:
#         error = np.stack([group[1] for group in group_avg_error])

#     return mean, error

def evoked_list(group_epoch_list):
    '''get average data from epoch list.
    Parameters:
    -----------------
    group_epoch_list: list, epoch objects to average over
    Output:
    ------------------
    group: list, ERP array for each participant in group/condition. elec x samples points
    '''
    
    group = []
    #loop through input to access averaged data for each participant
    for i in range(len(group_epoch_list)):
        evoked = group_epoch_list[i].average()
        avg_data = evoked.data
        group.append(avg_data)

    return group

def group_mean_error(listofarrays,error_type):
    '''Get ERP and error for group.
    Parameters:
    ---------------------------
    listofarrays: list, evoked arrays for single group/condition
    error_type: str, 'sem' or 'std'
    Output:
    --------------
    group_mean: numpy.ndarray, group ERP values. elec x sample points
    group_error: numpy.ndarray, group sem/std. elec x sample points
    '''
    #get group mean and group error
    group_array = np.stack(listofarrays,axis=0)
    group_mean = np.mean(group_array,axis=0)

    if error_type is None:
        return group_mean, None

    if error_type == 'sem':
        group_error = sem(group_array,axis=0)
    elif error_type == 'std':
        group_error = np.std(group_array,axis=0)

    return group_mean, group_error

def get_epoch_for_cond(all_data_epoch_list,cond):
    '''Get Epoch data for specific condition from Epoch list.
    Parameters:
    -----------------------
    all_data_epoch_list: list, nested list of epoch objects.
    cond: str, if single condition (i.e. 'SR')
          list, if multiple conditions are desired (i.e. ['SR','LR'])
    Output:
    -----------------------
    group_list: list, epoch data of single condition for single group
    output_data: tuple, list of epoch data for single condition for multiple groups
    cond_lists: tuple, list of epoch data for each desired condition. 
                If more than 1 group: 
                all conditions of group 1 listed first, then all conditions of group 2, etc.
    '''
    #single condition
    if type(cond) == str:
        #single group
        if len(all_data_epoch_list) == 1:
            group_list = [[]]
        
            for group in all_data_epoch_list:
                for part in group:
                    group_list[0].append(part[cond])
                
            return group_list
        #more than one group
        elif len(all_data_epoch_list) > 1:
            group_dict = {f'group_{group+1}': [] for group in range(len(all_data_epoch_list))}
        
        
            for groupidx,group in enumerate(all_data_epoch_list):
                for part in group:
                    group_dict[f'group_{groupidx+1}'].append(part[cond])
                
            output_data = [group_dict[f'group_{group+1}'] for group in range(len(all_data_epoch_list))]
        
            return tuple(output_data)
        
    #more than one condition
    elif type(cond) == list:
        cond_lists = []
        for groupidx, group in enumerate(all_data_epoch_list):
            for condition in cond:
                single_cond = []
                cond_lists.append(single_cond)
                for part in group:
                    single_cond.append(part[condition])
                    
        return tuple(cond_lists)

#####################################################################################################################################################
##################################################### Omni stat and other statistical functions #####################################################
#####################################################################################################################################################
#from scripts.classes import
#from scripts.signal import
# from scripts.stat import permutation_test, omni_stat
# from scripts.util import hyperbole_fit, print_dict, print_omni_dict
#from scripts.viz import

from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ranksums, wilcoxon, ttest_rel, ttest_ind

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from numpy import interp
from sklearn.metrics import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit
from scipy.stats import wilcoxon, ranksums, ttest_rel, ttest_ind, normaltest, bartlett
from sklearn.svm import SVC


def omni_stat(x, y, paired=False, tails=2, n_perm=int(1e5), verbose=0, effort=0):
    if paired:
        assert len(x) == len(y), 'Length of x and y is unequal. Choose paired=False instead!'
    p_perm = permutation_test(x, y, paired=paired, tails=tails, n_perm=n_perm, plot_me=False)

    # Test for normal distribution
    if len(x) >= 8 or len(y) >= 8:
        _, normality_x = normaltest(x)
        _, normality_y = normaltest(y)
    else:
        normality_x = normality_y = None

    if normality_x is not None:
        if normality_x > 0.05 and normality_y > 0.05:
            isnormal = ''
            normal = True
        else:
            isnormal = 'not '
            normal = False
    else:
        isnormal = 'maybe '
        normal = False
    p_welch = 42
    t_welch = 42
    # Test for homoscedasticity (equal variances):
    _, eq_var_p = bartlett(x, y)

    if eq_var_p > 0.05:
        isequal = ''
        equal = True
    else:
        isequal = 'not '
        equal = False
    if paired:
        if tails == -1:
            alternative = 'less'
        elif tails == 1:
            alternative = 'greater'
        elif tails == 2:
            alternative = 'two-sided'
        else:
            raise ValueError('tails must be -1, 1 or 2!')

        t_ttest, p_ttest = ttest_rel(x, y)
        t_wilc, p_wilc = wilcoxon(x, y, alternative=alternative)
    else:
        t_welch, p_welch = ttest_ind(x, y, equal_var=False)
        t_ttest, p_ttest = ttest_ind(x, y)
        t_wilc, p_wilc = ranksums(x, y)
    dec = decision_tree(equal, normal)
    # Cohens d
    d = cohens_d(x, y)
    # SVM discriminability
    loo = 0  # classification_test(x, y, effort=effort, cv='auto')

    omni_dict =  {'Is normal': normal,
            'Equal Variance': equal,
            'Rec': dec,
            'Mean Diff': np.mean(x) - np.mean(y),
            '': '',
            'p_perm': p_perm,
            ' ': '',
            'p_ttest': p_ttest,
            't_ttest': t_ttest,
            '  ': '',
            'p_wilc': p_wilc,
            't_wilc': t_wilc,
            '   ': '',
            'p_welch': p_welch,
            't_welch': t_welch,
            '    ': '',
            'Cohens d': d,
            'SVM acc': loo}
            
    if verbose == 1:
        print_omni_dict(omni_dict)
    return omni_dict


def print_dict(dct):
    ''' Print out a dictionary '''
    strings = ['Stats\n']
    for key, val in dct.items():
        try:
            keyvalpair = f'{key}: {val:.4}'
        except:
            keyvalpair = f'{key}: {val}'
        print(keyvalpair)
        strings.append(keyvalpair+'\n')
    return ''.join(strings)

def print_omni_dict(dct):
    strings = ['\nStats\n']
    for key, val in dct.items():
        if val != '':
            try:
                keyvalpair = f'{key}: {val:.4}'
            except:
                keyvalpair = f'{key}: {val}'
            print(keyvalpair)
            strings.append(keyvalpair+'\n')
        else:
            strings.append('\n')
    print('\n')
    return ''.join(strings)

def permutation_test(x, y, paired=False, n_perm=int(1e5), plot_me=False, tails=2):
    ''' performs a permutation test:
    paired... [True\False], whether the samples are paired
    n_perm... number of permutations defaults to 100k since numba makes it very fast :)
    plot_me... plot permuted values and true value
    tails... =2 (two-tailed), =1 (two-tailed & x>y) or =-1 (two-tailed and x<y)
    Author: Lukas Hecker, 03/2020
    '''
    if type(x) == list or type(y) == list:
        x = np.array(x)
        y = np.array(y)
    d_perm = np.zeros(shape=(n_perm))
    d_true = np.mean(x) - np.mean(y)
    if paired and len(x) == len(y):

        # assert len(x) == len(y), f"x and y are not of equal length, choose paired=False instead!\nx:{len(x)}, y:{len(y)}"

        signs = np.ones_like(x)
        signs[0:int(np.round(len(x)/2))] = signs[0:int(np.round(len(x)/2))] * -1
        diff = x-y
        d_perm = perm_pair(diff, n_perm, signs, d_perm)

    else:
        all_in_one_pot = np.array(list(x) + list(y))
        d_perm = perm_nopair(all_in_one_pot, n_perm, d_perm)


    # Calculate p as the proportion of permuted group differences that are abs() larger than the true difference:
    if tails == 2:
        p = len(np.where(np.abs(d_perm)>=np.abs(d_true))[0]) / n_perm
    elif tails == -1:
        p = len(np.where(d_perm<=d_true)[0]) / n_perm
    elif tails == 1:
        p = len(np.where(d_perm>=d_true)[0]) / n_perm
    else:
        raise NameError('tails should be either 2 (two-tailed), 1 (two-tailed & x>y) or -1 (two-tailed and y>x)')

    # Clip such that the lowest possible p-value is dependent onthe number of permutations
    p = np.clip(p, 1. / n_perm, None)

    if plot_me:
        plt.figure()
        plt.hist(d_perm)
        plt.plot([d_true, d_true], [plt.ylim()[0], plt.ylim()[1]])
        plt.title(f'd_true={d_true}, p={p}')

    return p

@jit(nopython=True)
def perm_nopair(all_in_one_pot, n_perm, d_perm):
    for i in range(n_perm):
        np.random.shuffle(all_in_one_pot)
        d_perm[i] = np.mean(all_in_one_pot[0:int(len(all_in_one_pot)/2)]) - np.mean(all_in_one_pot[int(len(all_in_one_pot)/2):])
    return d_perm

@jit(nopython=True)
def perm_pair(diff, n_perm, signs, d_perm):
    for i in range(n_perm):
        np.random.shuffle(signs)
        d_perm[i] = np.mean(diff * signs)
    return d_perm

###########################################################################################################################################
########################################### Plot Fkt Lukas --> Plot two ITV Evokeds with ERRORS ###########################################
###########################################################################################################################################

def plot_two_with_error(X, Y1, Y2, measure='SEM', labels=['Group A', 'Group B'],
        title='', ylabel='', xlabel='', xticklabels=None, test='perm', colors=None,
        paired=True):
    '''
    Plots two Grand mean traces with underlying error shading
    Parameters:
    -----------
    X : Time array
    Y1 : Group of observations 1
    Y2 : Group of observations 2
    measure : ['SEM', 'SD'] Choose standard error of the mean (SEM) or standard deviation (SD)
    test : ['perm', 'wilc', 'ttest']
    '''
    if type(Y1) == list:
        Y1 = np.squeeze(np.array(Y1))
        Y2 = np.squeeze(np.array(Y2))

    if colors is None:
        # colors = ['blue', 'orange']
        colors = ['#1f77b4', '#ff7f0e']


    m_y1 = np.mean(Y1, axis=0)
    sd_y1 = np.std(Y1, axis=0)
    # m_y1 = np.multiply(m_y1, 10**6)
    # sd_y1 = np.multiply(sd_y1, 10**6)


    m_y2 = np.mean(Y2, axis=0)
    sd_y2 = np.std(Y2, axis=0)
    # m_y2 = np.multiply(m_y2, 10**6)
    # sd_y2 = np.multiply(sd_y2, 10**6)

    if measure=='SEM':
        sd_y1 = np.array(sd_y1)
        sd_y1 /= np.sqrt(len(Y1))
        sd_y2 = np.array(sd_y2)
        sd_y2 /= np.sqrt(len(Y2))
    elif measure=='SD' or measure=='STD':
        print('')
    else:
        print(f'measure {measure} not available. Choose SEM or SD.')
        return
    plt.figure()
    ax1 = plt.subplot(211)
    if len(X) == 0:
        X = np.arange(len(m_y1))
    # Plot Y1
    plt.plot(X, m_y1, label=labels[0], color=colors[0])
    plt.fill_between(X, m_y1-sd_y1, m_y1+sd_y1, alpha=0.3, color=colors[0])
    # Plot Y2
    plt.plot(X, m_y2, label=labels[1], color=colors[1])
    plt.fill_between(X, m_y2-sd_y2, m_y2+sd_y2, alpha=0.3, color=colors[1])
    plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xticklabels != None:
        plt.xticks(X, xticklabels)
    plt.title(title)
    # P-values
    ax2 = plt.subplot(212)
    p_vals = np.zeros_like(m_y1)
    for i in range(len(p_vals)):
        if paired:
            if test == 'perm':
                p_vals[i] = permutation_test(Y1[:, i], Y2[:, i], paired=True)
            if test == 'wilc':
                p_vals[i] = wilcoxon(Y1[:, i], Y2[:, i])[1]
            if test == 'ttest':
                p_vals[i] = ttest_rel(Y1[:, i], Y2[:, i])[1]
        else:
            if test == 'perm':
                p_vals[i] = permutation_test(Y1[:, i], Y2[:, i])
            if test == 'wilc':
                p_vals[i] = ranksums(Y1[:, i], Y2[:, i])[1]
            if test == 'ttest':
                p_vals[i] = ttest_ind(Y1[:, i], Y2[:, i])[1]

    plt.plot(X, p_vals, color='dimgrey')  # , aspect='auto')
    plt.plot(X, np.ones_like(X)*0.05, '--', color='orangered')
    plt.ylabel("P-Value")
    plt.yscale('log')
    plt.title(f'{test} p-value')
    plt.xlabel("Time [s]")
    if xticklabels != None:
        plt.xticks(X, xticklabels)
    # Styling
    # if squareplot:
    #     square_axis(ax1)
    #     square_axis(ax2)

    plt.tight_layout(pad=1)

def decision_tree(equal, normal):
    if equal and normal:
        dec = ['t-tst']
    elif equal and (not normal):
        dec = ['wilc', 'perm']
    elif (not equal) and normal:
        dec = ['welch']
    elif (not equal) and (not normal):
        dec = ['perm']
    else:
        msg = f'this combination of equal={equal} and normal={normal} is not covered'
        raise ValueError(msg)
    return dec

def cohens_d(x, y):
    n1 = len(x)
    n2 = len(y)
    s = np.sqrt( ((n1 - 1) * variance(x) + (n2 - 1) * variance(y)) / (n1 + n2 - 2) )
    d = (np.mean(x) - np.mean(y)) / s
    return d

def variance(x):
    return np.sum((x - np.mean(x))**2) / (len(x)-1)

####################################################################################################################################################

def plot_hyperbole(ax):
    rc('font', family='sserif')
    x = np.arange(4, 100, 2)
    a, b, c = 30, 6.5, 0.4

    # fig = plt.figure(figsize=(4, 4))
    ax.plot(x, hyperbole_fit(x, a, b, c), color='black', linewidth=2)
    ax.set_title('Hyperbolic Function', fontsize=14)
    ax.text(30, 2.5, r'$f(x) = \left(\frac{A}{x + B}\right)+c$', fontsize=14)
    ax.set_yticks(np.arange(1, 4))
    ax.set_xticks([0, 50, 100])
    # return fig

def scatter_plot(x1, x2, title='', stat='wilc', paired=False, n_perm=int(1e6),
        legend=('Group A', 'Group B'), plot_stat_summary=True, effort=1):
    ''' This function takes two vectors and plots them as scatter
    plot and adds statistics to title'''

    if len(x1) > 0:
        if not x1[0]:
            return
    elif not x1:
        return

    if type(x1) == list:
        x1 = np.asarray(x1)
    if type(x2) == list:
        x2 = np.asarray(x2)

    val = [-2, 2]
    pos_x1 = val[0] + (np.random.rand(x1.size) * 2 - 1) * 0.3
    pos_x2 = val[1] + (np.random.rand(x2.size) * 2 - 1) * 0.3
    if stat=='wilc':
        if paired:
            t, p = wilcoxon(x1, x2)
        else:
            t, p = ranksums(x1, x2)
    elif stat=='ttest':
        if paired:
            t, p = ttest_rel(x1, x2)
        else:
            t, p = ttest_ind(x1, x2)
    elif stat == 'perm':
        p = permutation_test(x1, x2, paired=paired, tails=2, plot_me=False, n_perm=n_perm)
        t = 0

    plt.figure()
    colors = ['#0066ff', '#ff9900']
    plt.plot(pos_x1, x1, 'o', color=colors[0], markeredgecolor='black', label=legend[0])
    bbox_props = dict(facecolor=colors[0], alpha=0.6)
    plt.boxplot(x1, positions=[-2], patch_artist=True, notch=False, bootstrap=1000, whis=[5, 95], widths=0.5, boxprops=bbox_props, capprops=dict(color='black'), medianprops=dict(color='black', linewidth=2), showfliers=False)


    plt.plot(pos_x2, x2, 'o', color=colors[1], markeredgecolor='black', label=legend[1])
    bbox_props = dict(facecolor=colors[1], alpha=0.6)
    plt.boxplot(x2, positions=[2], patch_artist=True, notch=False, bootstrap=1000, whis=[5, 95], widths=0.5, boxprops=bbox_props, capprops=dict(color='black'), medianprops=dict(color='black', linewidth=2), showfliers=False)

    first_patch = mpatches.Patch(color=colors[0], label=legend[0])
    second_patch = mpatches.Patch(color=colors[1], label=legend[1])
    plt.legend(handles=[first_patch, second_patch])


    plt.title(f'{title}')

    if plot_stat_summary:
        om = omni_stat(x1, x2, paired=False, tails=2, verbose=0, effort=effort)
        txt = print_omni_dict(om)

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        pos_x = -5.9
        pos_y = np.max(np.concatenate((x1, x2), axis=0))
        plt.text(pos_x, pos_y, txt, fontsize=8, verticalalignment='top', bbox=props)
        plt.xlim((-6, 6))
    else:
        plt.xlim((-5, 5))

    plt.show()

def plot_roc(data, label, C, gamma, title):
    '''Function that uses cross validation to get an ROC curve of each fold, resulting in an average ROC curve.
    Uses the standard scaler function to normalise training data and apply the values found in the training set
    on the test set.
    input: data array(X), label array(y), SVM parameters C and gamma, and a string for the title'''
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.125, random_state=70)
    clf = SVC(kernel = 'rbf', C=C, gamma=gamma) #rbf kernel because it requires no background knowledge
    X = data
    y = label
    scaler = StandardScaler()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(sss.split(X, y)):
        #normalising the training data and applying the training sets mean and std on the testing set
        X[train] = scaler.fit_transform(X[train])
        X[test] = scaler.transform(X[test])

        clf.fit(X[train], y[train])

        viz = plot_roc_curve(clf, X[test], y[test],
                         name='Fold {}'.format(i+1),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orangered',
        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='mediumblue',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='cornflowerblue', alpha=.2, label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title= title)
    ax.set_aspect('equal', 'box')

    #new legend because of 100 folds make the legend above very big. and i dont know how to remove the legend in code above
    ax.get_legend().remove()

    custom_lines = [Line2D([0], [0], linestyle='--', color='orangered', lw=2),
                    Line2D([0], [0], color='mediumblue', lw=2),
                    Line2D([0], [0], color='cornflowerblue', lw=8,alpha=.2)]

    ax.legend(custom_lines,['Chance','Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),r'$\pm$ 1 std. dev.'],loc='center',bbox_to_anchor=(1.5, 0.45))
    #plt.show()