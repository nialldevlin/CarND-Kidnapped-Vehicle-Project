/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  std::cout << "Initialized" << std::endl << std::flush;

  num_particles = 100;	//Set number of particles

  std::normal_distribution<double> dist_x(x, std[0]);	//X distribution
  std::normal_distribution<double> dist_y(y, std[1]);	//Y distribution
  std::normal_distribution<double> dist_theta(theta, std[2]);	//Theta distribution
  
  particles.reserve(num_particles);	//Set size for faster particle loading
  weights.reserve(num_particles);		//Set size for faster weight loading

  for (int i = 0; i < num_particles; i++) {
    Particle part;
    
    part.id = i;	//Particle id is location in the vector
    part.x = dist_x(gen);		//Set x position with random gaussian noise
    part.y = dist_y(gen);		//Set y position with random gaussian noise
    part.theta = dist_theta(gen);		//Set theta position with random gaussian noise
    part.weight = 1.0;	//Set weight to 1
    weights.push_back(1.0);	//Add weights
    particles.push_back(part);		//Add particle
  }
  is_initialized = true;	//It is initialized
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  std::cout << "Prediction" << std::endl << std::flush;

  std::normal_distribution<double> dist_x(0.0, std_pos[0]);	//Random x gaussian noise
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);	//Random y guassian noise
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);	//Random theta gaussian noise
  
  for(Particle& particle : particles){
    double theta = particle.theta;
    
    if(std::abs(yaw_rate) < 0.00001){	//If yaw_rate zero use equations
      particle.x += velocity * delta_t * cos(theta);
      particle.y += velocity * delta_t * sin(theta);
      particle.theta = theta;
           
    } else {	//If yaw rate is not zero use equations
      particle.x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particle.y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t; 
    }
    particle.x += dist_x(gen);	//Add x noise
    particle.y += dist_y(gen);	//Add y noise
    particle.theta += dist_theta(gen);	//Add theta noise
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  std::cout << "Update Weights" << std::endl << std::flush;

  for(Particle& p : particles){
    
    //Convert each observation coordinates to map
    std::vector<LandmarkObs> mapObs;
    mapObs.reserve(observations.size());
    for(const LandmarkObs& o : observations){
      LandmarkObs obs;
      // transform to map x coordinate
      obs.x = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;

      // transform to map y coordinate
      obs.y = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;
      mapObs.push_back(obs);
      //std::cout << o.x << " : " << o.y << " , " << obs.x << " : " << obs.y << std::endl << std::flush;
    }
    
    //Find landmarks in sensor range
    std::vector<LandmarkObs> lm_in_range;
    lm_in_range.reserve(observations.size());
    for(const Map::single_landmark_s& l : map_landmarks.landmark_list){
      if(dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range){
        LandmarkObs lm;
        lm.x = l.x_f;
        lm.y = l.y_f;
        lm_in_range.push_back(lm);
      }
    }
    
    //Map observations to landmarks
    //dataAssociation(lm_in_range, mapObs, p);
    
    //Calculate weights with gaussian
    p.weight = 1.;
    weights[p.id] = 1.;
    for(const LandmarkObs& m : mapObs){   
      // calculate weight
      
      double mu_x;
      double mu_y;
      double curr_dist = 0;
      double prev_dist = std::numeric_limits<double>::infinity();
      
      for(LandmarkObs& lm : lm_in_range){
        curr_dist = dist(lm.x, lm.y, m.x, m.y);
        if(curr_dist < prev_dist){	//Loop until smallest distance, this is the closest
          prev_dist = curr_dist;
          mu_x = lm.x;
          mu_y = lm.y;
        }
      }      
      double weight = multiv_prob(std_landmark[0], std_landmark[1], m.x, m.y, mu_x, mu_y);
      if( weight > 0 ){
      	p.weight *= weight;
      }
      
    }
    weights[p.id] = p.weight;
  }
  std::cout << std::endl;
}

void ParticleFilter::resample() {
  std::cout << "Resampled" << std::endl << std::flush;
  std::vector<Particle> new_particles;
  new_particles.reserve(num_particles);	//New list of particles
  std::discrete_distribution<int> dist_w(weights.begin(), weights.end());	//Discrete distribution from weights
  for(int i = 0; i < num_particles; i++){
    int id = dist_w(gen);
    new_particles.push_back(particles[id]);	//select particles based on id picked from discrete distribution
  }
  particles = new_particles;	//replace old particles
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