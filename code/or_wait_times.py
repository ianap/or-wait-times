#! /usr/bin/env python

#
# mc
#
# Perform a Monte Carlo simulation of operating room waiting times
#
# Joe Antognini
# Tue Dec 24 15:01:48 EST 2013
#

#
# Program output -- each line of output consists of the following data:
#   -- Patient class (Emergent, Urgent 1, etc.)
#   -- The time the patient arrived.
#   -- Whether the patient arrived during the day or the night.
#   -- The time the patient had to wait
#   -- The time the patient spent in surgery
#

# Import the necessary Python packages so we have all the functions we need.
import random
import numpy
import datetime
import sys

def model_ors(n_day_oprooms, 
              n_classes, 
              distribution_parameters,
              n_night_oprooms=None, 
              night_length=8,
              converge_time=1e5):
  '''Run a Monte Carlo simulation of patients moving through a hospital's
  operating room system. 

  Parameters:
    n_day_oprooms: int
      The number of operating rooms during the day

    n_classes: int, optional
      The number of patient classes

    distribution_parameters: list or tuple
      A list or tuple describing the probability distribution of arrival
      times and surgery times.  This parameter must be as long as the number
      of classes (n_classes).  Each element must consist of a list or tuple
      of three elements.  The first describes the mean arrival rate, the
      second describes the mean surgery length, and the third describes the
      standard deviation of the surgery length.

    n_night_oprooms: int, optional
      The number of operating rooms during the night.  By default this is
      the same as the number of operating rooms during the day.

    night_length: float
      The number of hours per day that only night operating rooms are used.
    
    converge_time: float, optional
      The number of seconds to discard at the beginning of the program to
      let the program converge.
      
  Important data structures in this program:
    emergent_queue: This is a list of patients in the emergency queue.  Each
      patient is recorded in the list by the time that the patient arrived.
      The list is sorted in order of increasing time.  Thus the first
      patient in the queue is the patient who has been waiting the longest,
      and is the first patient who will be removed from the queue when an
      operating room frees up.
 
    urgent1_queue, urgent2_queue, etc.: Identical to emergent_queue, except
      for the other classes of patients.  These patients are served after
      the emergent patients.
 
    operating_rooms: This is a list of all operating rooms currently in use.
      Each operating room in use is recorded in this list by the time when
      the operating room will be available again. 
 
    utilization_frac: This is a list used to calculate how often the
      operating rooms are used.  The first element of this list is the total
      amount of time that exactly zero operating rooms have been in use,
      the second element is the total amount of time that exactly one
      operating room was in use, etc.  The utilization fraction is then
      calculated by taking these numbers and dividing by the total length of
      the experiment. 
  '''

  if n_night_oprooms is None:
    n_night_oprooms = n_day_oprooms

  # A few constants
  day_to_min = 1440 # Number of minutes in a day
  hour_to_min = 60 # Number of minutes in an hour

  # Typechecking
  if type(n_day_oprooms) is not int:
    raise TypeError('model_ors(): n_day_oprooms must be int!')
  elif n_day_oprooms <= 0:
    raise ValueError('model_ors(): n_day_oprooms must be positive!')
  if type(n_classes) is not int:
    raise TypeError('model_ors(): n_classes must be int!')
  elif n_classes <= 0:
    raise ValueError('model_ors(): n_classes must be positive!')
  if type(distribution_parameters) not in (list, tuple):
    raise TypeError('distribution_parameters must be tuple!')
  if len(distribution_parameters) != n_classes:
    raise ValueError(\
    'model_ors(): length of distribution_parameters not equal to n_classes!')
  for elem in distirbution_parameters:
    if type(elem) not in (list, tuple):
      raise TypeError('model_ors(): ' + str(elem) + 
      ' must be list or tuple')
    if len(elem) != 3:
      raise ValueError('model_ors(): ' + str(elem) + 
      ' must have exactly three elements!')
    for item in elem:
      if type(item) not in (int, float, long):
        raise TypeError('model_ors(): ' + str(item) + ' in ' + str(elem)
        + ' is not a number!')
      elif item < 0:
        raise ValueError('model_ors(): ' + str(item) + ' in ' + str(elem)
        + ' must be positive!')
    if n_night_oprooms is not None:
      if type(n_night_oprooms) is not int:
        raise TypeError('model_ors(): n_night_oprooms must be int!')
      elif n_night_oprooms > n_day_oprooms:
        raise ValueError(\
        'model_ors(): n_night_oprooms must be less than n_day_oprooms!')
      elif n_night_oprooms < 0:
        raise ValueError(\
        'model_ors(): n_night_oprooms must be non-negative!')
    if type(night_length) not in (int, float, long):
      raise TypeError('model_ors(): night_length must be a number!')
    elif night_length < 0:
      raise ValueError('model_ors(): night_length must be non-negative!')
    if type(converge_time) not in (int, float, long):
      raise TypeError('model_ors(): converge_time must be a number!')
    elif converge_time < 0:
      raise ValueError('model_ors(): converge_time must be positive!')
        
# These lines read in the number of operating rooms that the user has
# supplied.
n_day_oprooms = int(sys.argv[1])
n_night_oprooms = int(sys.argv[2])

# The number of ORs during the day should be larger than the number of ORs
# during the night.  Make sure that the larger number of ORs is saved as the
# number of ORs during the day.
if n_day_oprooms < n_night_oprooms:
  n_day_oprooms, n_night_oprooms = (n_night_oprooms, n_day_oprooms)

# These numbers describe the observed probabilities of for patient arrival
# time rates and surgery times.  See the notebook for the derivation of
# these numbers.

# The arrival time is described approximately by a Poisson distribution
p_in_emergent = .001607686 # per minute
p_in_urgent1 = .003232496
p_in_urgent2 = .002665525
p_in_urgent3 = .000334855
p_in_urgent4 = .00
p_in_AE = .001288052

# The surgery time is described well by a log-normal distribution
mean_surg_time_emergent = 5.00716
mean_surg_time_urgent1 = 4.96477
mean_surg_time_urgent2 = 5.05842
mean_surg_time_urgent3 = 5.00069
mean_surg_time_urgent4 = 4.95519
mean_surg_time_AE = 5.01655

sigma_surg_time_emergent = .583642
sigma_surg_time_urgent1 = .677607
sigma_surg_time_urgent2 = .651279
sigma_surg_time_urgent3 = .570812
sigma_surg_time_urgent4 = .665382
sigma_surg_time_AE  = .713405

def is_day(time, night_hours=8):
  '''This function returns True if it is day and False if it is night.  We
  assume that night lasts eight hours unless the user specifies
  otherwise.'''

  if time % day_to_min > night_hours * hour_to_min:
    return True
  else:
    return False

# We will run the simulation for a little while without printing any data
# to wait for it to converge.
converge_time = 1e5
#experiment_length = int(2.628e6)
experiment_length = int(2e5)
time = 0

# These lines initialize the queues to be empty.
operating_rooms = []
emergent_queue = []
urgent1_queue = []
urgent2_queue = []
urgent3_queue = []
urgent4_queue = []
AE_queue = []
utilization_frac = (n_day_oprooms + 1) * [0]

# Now we start at time 0, and step through minute by minute.
for i in range(experiment_length):
  # First we see how many patients have arrived this minute.
  n_emergent_patients = numpy.random.poisson(p_in_emergent)
  n_urgent1_patients = numpy.random.poisson(p_in_urgent1)
  n_urgent2_patients = numpy.random.poisson(p_in_urgent2)
  n_urgent3_patients = numpy.random.poisson(p_in_urgent3)
  n_urgent4_patients = numpy.random.poisson(p_in_urgent4)
  n_AE_patients = numpy.random.poisson(p_in_AE)

  # If any patients have arrived, add them to the queue.
  for j in range(n_emergent_patients):
    emergent_queue.append(time)
  for j in range(n_urgent1_patients):
    urgent1_queue.append(time)
  for j in range(n_urgent2_patients):
    urgent2_queue.append(time)
  for j in range(n_urgent3_patients):
    urgent3_queue.append(time)
  for j in range(n_urgent4_patients):
    urgent4_queue.append(time)
  for j in range(n_AE_patients):
    AE_queue.append(time)  

  # Check to see if any operating rooms have opened up this minute.
  operating_rooms = [x for x in operating_rooms if x != time]

  # Change the number of operating rooms based on the time of day.
  # Currently the program is set up to assume that night lasts eight hours.
  if is_day(time):
    n_oprooms = n_day_oprooms
  else:
    n_oprooms = n_night_oprooms

  # Check to see if there are any open operating rooms and any patients in
  # the emergency queue.
  if len(operating_rooms) < n_oprooms and len(emergent_queue) > 0:
    # If a patient from the emergency queue can be put in an operating room
    # this minute, calculate the time the patient had to wait.
    wait_time = time - emergent_queue[0]

    # Draw the amount of time that the patient has to spend in surgery from
    # a log-normal distribution.
    surgery_time = int(round(numpy.random.lognormal(mean_surg_time_emergent, sigma_surg_time_emergent)))

    # Print the data about this patient.
    if time > converge_time:
      if is_day(emergent_queue[0]):
        time_of_day = "day"
      else:
        time_of_day = "night"
      print "emergent", emergent_queue[0], time_of_day, wait_time, \
        surgery_time

    # Because the patient has been put in an operating room, he can be
    # removed from the emergency queue.
    emergent_queue.remove(emergent_queue[0])

    # 60 minutes is added to the surgery time for clean up.
    out_time = time + surgery_time + 60 
    
    # Lastly, record the time that the operating room will be free in the
    # operating rooms list. 
    operating_rooms.append(out_time)

  if len(operating_rooms) < n_oprooms and len(urgent1_queue) > 0:
    wait_time = time - urgent1_queue[0]
    surgery_time = int(round(numpy.random.lognormal(mean_surg_time_urgent1, sigma_surg_time_urgent1)))
    if time > converge_time:
      if is_day(urgent1_queue[0]):
        time_of_day = "day"
      else:
        time_of_day = "night"
      print "urgent1", urgent1_queue[0], time_of_day, wait_time, surgery_time
  
    urgent1_queue.remove(urgent1_queue[0])
  
    # 60 minutes is added to the surgery time for clean up.
    out_time = time + surgery_time + 60 
    operating_rooms.append(out_time)
  
  if (len(operating_rooms) < n_oprooms and len(urgent2_queue) and
    is_day(time)) > 0:
    wait_time = time - urgent2_queue[0]
    surgery_time = int(round(numpy.random.lognormal(mean_surg_time_urgent2, sigma_surg_time_urgent2)))
    if time > converge_time:
      if is_day(urgent2_queue[0]):
        time_of_day = "day"
      else:
        time_of_day = "night"
      print "urgent2", urgent2_queue[0], time_of_day, wait_time, surgery_time
  
    urgent2_queue.remove(urgent2_queue[0])
  
    # 60 minutes is added to the surgery time for clean up.
    out_time = time + surgery_time + 60 
    operating_rooms.append(out_time)

  if (len(operating_rooms) < n_oprooms and len(urgent3_queue) and
    is_day(time)) > 0:
    wait_time = time - urgent3_queue[0]
    surgery_time = int(round(numpy.random.lognormal(mean_surg_time_urgent3, sigma_surg_time_urgent3)))
    if time > converge_time:
      if is_day(urgent3_queue[0]):
        time_of_day = "day"
      else:
        time_of_day = "night"
      print "urgent3", urgent3_queue[0], time_of_day, wait_time, surgery_time
  
    urgent3_queue.remove(urgent3_queue[0])
  
    # 60 minutes is added to the surgery time for clean up.
    out_time = time + surgery_time + 60 
    operating_rooms.append(out_time)

  if (len(operating_rooms) < n_oprooms and len(urgent4_queue) and
    is_day(time))> 0:
    wait_time = time - urgent4_queue[0]
    surgery_time = int(round(numpy.random.lognormal(mean_surg_time_urgent4, sigma_surg_time_urgent4)))
    if time > converge_time:
      if is_day(urgent4_queue[0]):
        time_of_day = "day"
      else:
        time_of_day = "night"
      print "urgent4", urgent4_queue[0], time_of_day, wait_time, surgery_time
  
    urgent4_queue.remove(urgent4_queue[0])
  
    # 60 minutes is added to the surgery time for clean up.
    out_time = time + surgery_time + 60 
    operating_rooms.append(out_time)
  
  if (len(operating_rooms) < n_oprooms and len(AE_queue) and is_day(time)) > 0:
    wait_time = time - AE_queue[0]
    surgery_time = int(round(numpy.random.lognormal(mean_surg_time_AE, sigma_surg_time_AE)))
    if time > converge_time:
      if is_day(AE_queue[0]):
        time_of_day = "day"
      else:
        time_of_day = "night"
      print "AE", AE_queue[0], time_of_day, wait_time, surgery_time
  
    AE_queue.remove(AE_queue[0])
  
    # 60 minutes is added to the surgery time for clean up.
    out_time = time + surgery_time + 60 
    operating_rooms.append(out_time)

  if time > converge_time:
    utilization_frac[len(operating_rooms)] += 1
  time += 1

i = 0
for elem in utilization_frac:
  print >> sys.stderr, i, "/", n_oprooms, float(elem) / (experiment_length - converge_time)
  i += 1

# P.S. Valerie says hi!
