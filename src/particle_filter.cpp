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

	std::normal_distribution<double> distX(x, std[0]);
	std::normal_distribution<double> distY(y, std[1]);
	std::normal_distribution<double> distT(theta, std[2]);

	particles.reserve(num_particles);
	weights.reserve(num_particles);

	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = distX(gen);
		p.y = distY(gen);
		p.theta = distT(gen);
		p.weight = 1.0;

		weights.push_back(1.0);
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
								double velocity, double yaw_rate) {
	std::normal_distribution<double> distX(0, std_pos[0]);
	std::normal_distribution<double> distY(0, std_pos[1]);
	std::normal_distribution<double> distT(0, std_pos[2]);

	for(Particle& p : particles) {
		double x = p.x;
		double y = p.y;
		double theta = p.theta;

		if(std::abs(yaw_rate) > 0.0001) {
			p.x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
			p.y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
			p.theta = theta + yaw_rate * delta_t;
		} else {
			p.x = x + velocity * delta_t * cos(theta);
			p.y = y + velocity * delta_t * sin(theta);
			p.theta = theta;
		}

		p.x += distX(gen);
		p.y += distY(gen);
		p.theta += distT(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
                                     std::vector<LandmarkObs>& observations) {
	double curr_dist;
	double prev_dist;
	for (LandmarkObs& obs : observations) {

		prev_dist = std::numeric_limits<double>::max();

		for (LandmarkObs& p : predicted) {

			curr_dist = dist(obs.x, obs.y, p.x, p.y);

			if (prev_dist > curr_dist) {
				obs.id = p.id;
				prev_dist = curr_dist;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  double weight;
  double exponent;
  
  for (Particle& p : particles) {
      std::vector<LandmarkObs> map_obs;
      
      for (LandmarkObs obs : observations) {
          LandmarkObs o;
          
          o.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
          o.y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
          o.id = obs.id;

          map_obs.push_back(o);
        }
        
        std::vector<LandmarkObs> pred_landmarks;

      for (unsigned int ld_index = 0; ld_index < map_landmarks.landmark_list.size(); ld_index++) 
        {
          int ld_id = map_landmarks.landmark_list[ld_index].id_i;
          double ld_x = map_landmarks.landmark_list[ld_index].x_f;
          double ld_y = map_landmarks.landmark_list[ld_index].y_f;

          double landMarkDistFromParticle = dist(p.x, p.y, ld_x, ld_y);

          if (landMarkDistFromParticle <= sensor_range) 
          {
            LandmarkObs l_is_within_range;
            l_is_within_range.id = ld_id;
            l_is_within_range.x = ld_x;
            l_is_within_range.y = ld_y;
            pred_landmarks.push_back(l_is_within_range);
          }
        }
    
      dataAssociation(pred_landmarks, map_obs);
      
	  weight = 1.0;
      p.weight = 1.0;
      
      double landmark_x, landmark_y;
      double var_x = pow(std_landmark[0], 2);
      double var_y = pow(std_landmark[1], 2);

      for (unsigned int i = 0; i < map_obs.size(); i++) 
      {
        for (unsigned int ld_indx = 0; ld_indx < pred_landmarks.size(); ld_indx++) 
        {
          if (map_obs[i].id == pred_landmarks[ld_indx].id) 
          {
            landmark_x = pred_landmarks[ld_indx].x;
            landmark_y = pred_landmarks[ld_indx].y;
            exponent = (pow(map_obs[i].x - landmark_x, 2) / (2 * var_x)) + (pow(map_obs[i].y - landmark_y, 2) / (2 * var_y));
            //std::cout << "probability " << probability << std::endl;
            weight = weight * gauss_norm * exp(-exponent);
            break;
          }
        }
      }
    std::cout << "weight: " << weight << std::endl;   
    p.weight = weight;
    weights[p.id] = weight;
  }
  double sumOfWeights = 0.0;
  for (int p_i = 0; p_i < num_particles; p_i++) 
  {
    sumOfWeights += particles[p_i].weight;
  }
  std::cout << "normalizing constant " << sumOfWeights << std::endl;
  if (sumOfWeights == 0) {
  	exit(3);
  }
  for (int p_i = 0; p_i < num_particles; p_i++) 
  {
    particles[p_i].weight = particles[p_i].weight / sumOfWeights;
    weights[p_i] = particles[p_i].weight;
  }
}

void ParticleFilter::resample() {
	std::discrete_distribution<int> dist_w(weights.begin(), weights.end());

	std::vector<Particle> new_particles;
	new_particles.reserve(num_particles);

	for (int i = 0; i < num_particles; i++) {
		int index = dist_w(gen);
		new_particles.push_back(particles[index]);
	}

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