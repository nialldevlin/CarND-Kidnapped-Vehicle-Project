#include "particle_filter.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <stdlib.h>

#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 500;

	//Create gaussian distributions to add random noise
	std::normal_distribution<double> distX(x, std[0]);
	std::normal_distribution<double> distY(y, std[1]);
	std::normal_distribution<double> distT(theta, std[2]);

	//Reserve space
	particles.reserve(num_particles);
	weights.reserve(num_particles);

	//Initialize each particle
	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;	//Keep track of particles
		p.x = distX(gen);	//Add random noise
		p.y = distY(gen);	
		p.theta = distT(gen);
		p.weight = 1.0;	//Default weight is zero

		weights.push_back(1.0);
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
								double velocity, double yaw_rate) {

	//Create random noise
	//This only needs to be done once
	std::normal_distribution<double> distX(0, std_pos[0]);
	std::normal_distribution<double> distY(0, std_pos[1]);
	std::normal_distribution<double> distT(0, std_pos[2]);

	for(Particle& p : particles) {

		if(std::abs(yaw_rate) > 0.0001) {	//If yaw rate not zero update location
			p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta = p.theta + yaw_rate * delta_t;
		} else {	//If yaw rate is zero update location with different equations
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
			p.theta = p.theta;
		}

		//Add random noise;
		p.x += distX(gen);
		p.y += distY(gen);
		p.theta += distT(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
                                     std::vector<LandmarkObs>& observations) {
	double curr_dist;
	double prev_dist;

	//Loop through both vectors
	for (LandmarkObs& obs : observations) {

		prev_dist = std::numeric_limits<double>::max();

		for (LandmarkObs& p : predicted) {

			curr_dist = dist(obs.x, obs.y, p.x, p.y);

			if (prev_dist > curr_dist) {	//Find lowest distance and assign ID to match
				obs.id = p.id;
				prev_dist = curr_dist;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);	//Only need to calculate this once
  double weight;
  double exponent;
  double sum = 0;	//Normalize all weights at the end
  
  for (Particle& p : particles) {
  	//Create vector of observations in map reference frame
    std::vector<LandmarkObs> map_obs;
    map_obs.reserve(observations.size());
      
    for (LandmarkObs obs : observations) {
      LandmarkObs o;
      
      //Convert coordinates to map frame
      o.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      o.y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
      o.id = obs.id;

      map_obs.push_back(o);
    }
    
    //Create vector of landmarks in sensor range
    std::vector<LandmarkObs> lm_in_range;
    lm_in_range.reserve(observations.size());
	double d;
    
    for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {

      d = dist(p.x, p.y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);

      //If distance to particle is withing sensor range it is a valid point
      if (d <= sensor_range) {
        LandmarkObs l;
        l.id = map_landmarks.landmark_list[i].id_i;
        l.x = map_landmarks.landmark_list[i].x_f;
        l.y = map_landmarks.landmark_list[i].y_f;
        lm_in_range.push_back(l);
      }
    }
    
    //Correlate observations and landmarks
    dataAssociation(lm_in_range, map_obs);
    
    //Reset
	weight = 1.0;
    p.weight = 1.0;
    
    //Not the fastest way to do it
    //Caluclates weight based on each observation-landmark pair
    for (LandmarkObs& obs : map_obs) {
      for (LandmarkObs& lm : lm_in_range) {
        if (obs.id == lm.id) {
          exponent = (pow(obs.x - lm.x, 2) / (2 * pow(std_landmark[0], 2))) + (pow(obs.y - lm.y, 2) / (2 * pow(std_landmark[1], 2)));
          weight = weight * gauss_norm * exp(-exponent);
          break;
        }
      }
    }

    //Update weights
    p.weight = weight;
    weights[p.id] = weight;

    //Update sum
    sum += weight;
  }

  //Normalize, guard against division by zero
  if (sum != 0) {
  	for (int i = 0; i < num_particles; i++) {
      particles[i].weight /= sum;
      weights[i] = particles[i].weight;
    }
  }
}

void ParticleFilter::resample() {
	//Create distribution to pick the weights
	std::discrete_distribution<int> dist_w(weights.begin(), weights.end());

	//Duplicate empty particle array
	std::vector<Particle> new_particles;
	new_particles.reserve(num_particles);

	//Select new particles based on weight
	for (int i = 0; i < num_particles; i++) {
		int index = dist_w(gen);
		new_particles.push_back(particles[index]);
	}

	//Assign particles to old vector
	particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, 
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}