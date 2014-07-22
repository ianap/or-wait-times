#! /usr/bin/env python

import numpy
import sys

def model_ors(n_day_oprooms, 
              distribution_parameters,
              n_night_oprooms=None, 
              min_dayonly_class=3,
              night_length=8,
              converge_time=1e5,
              experiment_length=2e6,
              cleaning_time=60):
  '''Run a Monte Carlo simulation of patients moving through a hospital's
  operating room system. 

  Parameters:
    n_day_oprooms: int
      The number of operating rooms during the day

    distribution_parameters: list or tuple
      A list or tuple describing the probability distribution of arrival
      times and surgery times.  This parameter must be as long as the number
      of classes.  Each element must consist of a list or tuple of three
      elements.  The first describes the mean arrival rate, the second
      describes the mean surgery length, and the third describes the
      standard deviation of the surgery length.

    n_night_oprooms: int, optional
      The number of operating rooms during the night.  By default this is
      the same as the number of operating rooms during the day.

    min_dayonly_class: int, optional
      The lowest class that only enters surgery during the day.

    night_length: float
      The number of hours per day that only night operating rooms are used.
    
    converge_time: float, optional
      The number of seconds to discard at the beginning of the program to
      let the program converge.

    experiment_length: number, optional
      The number of seconds to run the experiment (in addition to the
      seconds discarded after converge_time)

    cleaning_time: number, optional
      The number of minutes before and after a surgery that the operating
      room is unavailable because it is being prepared or cleaned.
      
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
  hour_to_min = 60  # Number of minutes in an hour

  ###
  ### Typechecking
  ###

  if type(n_day_oprooms) is not int:
    raise TypeError('model_ors(): n_day_oprooms must be int!')
  elif n_day_oprooms <= 0:
    raise ValueError('model_ors(): n_day_oprooms must be positive!')
  if type(distribution_parameters) not in (list, tuple):
    raise TypeError('distribution_parameters must be tuple!')
  for elem in distribution_parameters:
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
  if type(experiment_length) not in (int, float, long):
    raise TypeError('model_ors(): experiment_length must be a number!')
  elif experiment_length < 0:
    raise ValueError('model_ors(): experiment_length must be positive!')
  if type(cleaning_time) not in (int, float, long):
    raise TypeError('model_ors(): cleaning_time must be a number!')
  elif cleaning_time < 0:
    raise ValueError('model_ors(): cleaning_time must be positive!')
      
  ###
  ### Done typechecking
  ### 

  n_classes = len(distribution_parameters)

  # These lines initialize the queues to be empty.
  operating_rooms = []
  queues = [[] for i in range(n_classes)]
  utilization_frac = (n_day_oprooms + 1) * [0]
  results = []

  # Now we start at time 0, and step through minute by minute.
  for time in range(int(experiment_length + converge_time)):
    # First we see how many patients have arrived this minute.
    n_patients = [numpy.random.poisson(x[0]) for x in
      distribution_parameters]

    # If any patients have arrived, add them to the queue.
    for j in range(n_classes):
      for k in range(n_patients[j]):
        queues[j].append(time)

    # Check to see if any operating rooms have opened up this minute.
    operating_rooms = [x for x in operating_rooms if x != time]

    # Change the number of operating rooms based on the time of day.
    # Currently the program is set up to assume that night lasts eight hours.
    if time % day_to_min > night_length * hour_to_min:
      n_oprooms = n_day_oprooms
    else:
      n_oprooms = n_night_oprooms

    for j in range(n_classes):
      # Check to see if there are any open operating rooms and any patients in
      # the emergency queue.
      if len(operating_rooms) < n_oprooms and len(queues[j]) > 0:
        # If a patient from the emergency queue can be put in an operating room
        # this minute, calculate the time the patient had to wait.
        wait_time = time - queues[j][0]

        # Draw the amount of time that the patient has to spend in surgery from
        # a log-normal distribution.
        surgery_time = int(round(numpy.random.lognormal(
          distribution_parameters[j][1], distribution_parameters[j][2])))

        # Print the data about this patient.
        if queues[j][0] > converge_time:
          if queues[j][0] % day_to_min > night_length * hour_to_min:
            time_of_day = "day"
          else:
            time_of_day = "night"
          results.append([j, int(queues[j][0] - converge_time), time_of_day, 
            wait_time, surgery_time])

        # Because the patient has been put in an operating room, he can be
        # removed from the emergency queue.
        queues[j].remove(queues[j][0])

        # 60 minutes is added to the surgery time for clean up.
        out_time = time + surgery_time + cleaning_time
        
        # Lastly, record the time that the operating room will be free in the
        # operating rooms list. 
        operating_rooms.append(out_time)

    if time > converge_time:
      utilization_frac[len(operating_rooms)] += 1

  utilization_results = []
  for i, elem in enumerate(utilization_frac):
    utilization_results.append([str(i) + '/' + str(n_oprooms), float(elem) / 
      (experiment_length - converge_time)])

  return (results, utilization_results)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print "usage: or_wait_times.py n_oprooms"
    sys.exit(1)

  # These lines read in the number of operating rooms that the user has
  # supplied.
  N_DAY_OPROOMS = int(sys.argv[1])

  # These numbers describe the observed probabilities of for patient arrival
  # time rates and surgery times.  See the notebook for the derivation of
  # these numbers.
  DISTRIBUTION_PARAMETERS = ((.001607686, 5.00716, .583642),
                             (.003232496, 4.96477, .677607),
                             (.002665525, 5.05842, .651279),
                             (.000334855, 5.00069, .570812),
                             (.00,        4.95519, .665382),
                             (.001288052, 5.01655, .713405))

  RESULTS, UTILIZATION_RESULTS = model_ors(N_DAY_OPROOMS,
    DISTRIBUTION_PARAMETERS, experiment_length=2e5)

  for elem in RESULTS:
    for item in elem:
      print item,
    print

  for elem in UTILIZATION_RESULTS:
    for item in elem:
      print >> sys.stderr, item,
    print >> sys.stderr
