/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LAYER_HH__
#define __LAYER_HH__

#include <fcntl.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include "CSVWriter.hh"
#include "Workload.hh"
#include "astra-sim/system/StreamStat.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {
class DataSet;
class Layer : public Callable, public StreamStat {
 public:
  std::string id;
  int layer_num;
  Sys* generator;
  Workload* workload;

  int fwd_pass_compute_time;
  ComType fwd_pass_comm_type;
  int fwd_pass_comm_size;
  Tick fwd_update_time;
  std::vector<bool> fwd_pass_comm_involved_dimensions;

  int input_grad_compute_time;
  ComType input_grad_comm_type;
  int input_grad_comm_size;
  Tick input_grad_update_time;
  std::vector<bool> input_grad_comm_involved_dimensions;

  int weight_grad_compute_time;
  ComType weight_grad_comm_type;
  int weight_grad_comm_size;
  Tick weight_grad_update_time;
  std::vector<bool> weight_grad_comm_involved_dimensions;

  bool needs_fwd_in_bckwd_initiation;
  bool is_checkpoint;
  ParallelismPolicy specific_parallellism;

  int lookup_table_size;
  int collective_counter;

  // DataSet *fwd_pass_dataset;
  // DataSet *input_grad_dataset;
  // DataSet *weight_grad_dataset;

  std::map<int, DataSet*> fwd_pass_datasets;
  std::list<Tick> started_waiting_for_fwd_pass;
  std::map<int, DataSet*> input_grad_datasets;
  std::list<Tick> started_waiting_for_input_grad;
  std::map<int, DataSet*> weight_grad_datasets;
  std::list<Tick> started_waiting_for_weight_grad;

  // reports
  Tick total_forward_pass_compute;
  Tick total_input_grad_compute;
  Tick total_weight_grad_compute;
  Tick total_weight_grad_comm;
  Tick total_input_grad_comm;
  Tick total_fwd_comm;

  Tick last_fwd_finished;
  Tick last_wg_finished;
  Tick last_ig_finished;

  Tick total_waiting_for_wg_comm;
  Tick total_waiting_for_ig_comm;
  Tick total_waiting_for_fwd_comm;

  CollectiveBarrier fwd_barrier;
  CollectiveBarrier wg_barrier;
  CollectiveBarrier ig_barrier;

  Layer(
      std::string id,
      int layer_num,
      Sys* generator,
      Workload* workload,
      int fwd_pass_compute_time,
      ComType fwd_pass_comm_type,
      int fwd_pass_comm_size,
      std::vector<bool> fwd_pass_comm_involved_dimensions,
      int input_grad_compute_time,
      ComType input_grad_comm_type,
      int input_grad_comm_size,
      std::vector<bool> input_grad_comm_involved_dimensions,
      int weight_grad_compute_time,
      ComType weight_grad_comm_type,
      int weight_grad_comm_size,
      std::vector<bool> weight_grad_comm_involved_dimensions,
      int weight_grad_update_time,
      ParallelismPolicy specific_policy);
  void call(EventType event, CallData* mdata);
  Tick get_fwd_pass_compute();
  Tick get_input_grad_compute();
  Tick get_weight_grad_compute();
  void increment_waiting_for_wg();
  void increment_waiting_for_ig();
  void increment_waiting_for_fwd();
  bool is_fwd_pass_comm_finished();
  bool is_input_grad_comm_finished();
  bool is_weight_grad_comm_finished();
  bool is_fwd_pass_comm_finished_blocking();
  bool is_input_grad_comm_finished_blocking();
  bool is_weight_grad_comm_finished_blocking();
  LayerData report(
      std::string run_name,
      int layer_num,
      int total_rows,
      int stat_row,
      CSVWriter* detailed,
      CSVWriter* EndToEnd,
      double& total_compute,
      double& total_exposed,
      bool seprate_log);
  void issue_forward_pass_comm(
      SchedulingPolicy pref_scheduling,
      CollectiveBarrier barrier);
  void issue_input_grad_comm(
      SchedulingPolicy pref_scheduling,
      CollectiveBarrier barrier);
  void issue_weight_grad_comm(
      SchedulingPolicy pref_scheduling,
      CollectiveBarrier barrier);
};
} // namespace AstraSim
#endif
