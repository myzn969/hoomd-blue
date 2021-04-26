// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "Autotuner.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cfloat>
#include <sstream>

#include <pybind11/stl.h>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

using namespace std;
namespace py = pybind11;


/*! \file Autotuner.cc
    \brief Definition of Autotuner
*/

/*! \param parameters List of valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration
*/
Autotuner::Autotuner(const std::vector<unsigned int>& parameters,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name),
      m_state(STARTUP), m_current_sample(0),  m_calls(0),
      m_exec_conf(exec_conf), m_mode(mode_median),
      m_attached(false), m_have_param(false), m_have_timing(false), m_cur_params_on_device(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << nsamples << " " << period << " " << name << endl;

    // initialize with a single dimension
    std::vector<std::vector<unsigned int> > params;
    params.push_back(parameters);
    initialize(params);

    // create CUDA events
    #ifdef ENABLE_HIP
    hipEventCreate(&m_start);
    hipEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif

    m_sync = false;
    }

/*! \param parameters List of valid parameters (n dimensions)
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration
*/
Autotuner::Autotuner(const std::vector< std::vector<unsigned int> >& parameters,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name),
      m_state(STARTUP), m_current_sample(0), m_calls(0),
      m_exec_conf(exec_conf), m_mode(mode_median),
      m_attached(false), m_have_param(false), m_have_timing(false), m_cur_params_on_device(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << nsamples << " " << period << " " << name << endl;

    initialize(parameters);

    // create CUDA events
    #ifdef ENABLE_HIP
    hipEventCreate(&m_start);
    hipEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif

    m_sync = false;
    }

/*! \param start first valid parameter
    \param end last valid parameter
    \param step spacing between valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration

    \post Valid parameters will be generated with a spacing of \a step in the range [start,end] inclusive.
*/
Autotuner::Autotuner(unsigned int start,
                     unsigned int end,
                     unsigned int step,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name),
      m_state(STARTUP), m_current_sample(0), m_calls(0),
      m_current_param(0), m_exec_conf(exec_conf), m_mode(mode_median), m_attached(false),
      m_have_param(false), m_have_timing(false), m_cur_params_on_device(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << " " << start << " " << end << " " << step << " "
                                << nsamples << " " << period << " " << name << endl;

    // initialize a flat autotuner with a range of parameters
    std::vector<std::vector< unsigned int > > params(1);
    params[0].resize((end - start) / step + 1);
    unsigned int cur_param = start;
    for (unsigned int i = 0; i < params[0].size(); i++)
        {
        params[0][i] = cur_param;
        cur_param += step;
        }
    initialize(params);

    // create CUDA events
    #ifdef ENABLE_HIP
    hipEventCreate(&m_start);
    hipEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif

    m_sync = false;
    }

Autotuner::~Autotuner()
    {
    m_exec_conf->msg->notice(5) << "Destroying Autotuner " << m_name << endl;
    #ifdef ENABLE_HIP
    hipEventDestroy(m_start);
    hipEventDestroy(m_stop);
    CHECK_CUDA_ERROR();
    hipFree(d_params);
    CHECK_CUDA_ERROR();
    #endif
    }

void Autotuner::initialize(const std::vector<std::vector<unsigned int> >& params)
    {
    m_parameters = params;

    // ensure that m_nsamples is odd (so the median is easy to get). This also ensures that m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    m_enable_dim.resize(m_parameters.size(), true);
    m_current_element.resize(m_parameters.size(), 0);
    m_current_param.resize(m_parameters.size());
    for (unsigned int n = 0; n < m_parameters.size(); ++n)
        {
        // initialize memory
        if (m_parameters[n].size() == 0)
            {
            this->m_exec_conf->msg->error() << "Autotuner " << m_name << ", dimension " << n << " got no parameters" << endl;
            throw std::runtime_error("Error initializing autotuner");
            }

        m_current_param[n] = m_parameters[n][m_current_element[n]];
        }

    #ifdef ENABLE_HIP
    hipMallocManaged(&d_params, sizeof(unsigned int)*m_parameters.size());
    CHECK_CUDA_ERROR();
    #endif
    }

/*! \param enabled true to enable sampling, false to disable it
    \param dimension to enable/disable
*/
void Autotuner::setEnableDimension(unsigned int dim, bool enabled)
    {
    assert(dim < m_enable_dim.size());
    m_enable_dim[dim] = enabled;
    }

void Autotuner::setEnabled(bool enabled)
    {
    m_enabled = enabled;

    if (!enabled)
        {
        m_exec_conf->msg->notice(6) << "Disable Autotuner " << m_name << std::endl;

        if (!m_attached && !m_have_param)
            {
            // if not complete, issue a warning
            if (!isComplete())
                {
                m_exec_conf->msg->notice(2) << "Disabling Autotuner " << m_name << " before initial scan completed!" << std::endl;
                }
            else
                {
                // ensure that we are in the idle state and have an up to date optimal parameter
                for (unsigned int i = 0; i < m_current_element.size(); ++i)
                    {
                    if (m_enable_dim[i])
                        {
                        m_current_element[i] = 0;
                        }
                    }
                m_state = IDLE;
                m_current_param = computeOptimalParameter();
                }
            }
        }
    else
        {
        m_exec_conf->msg->notice(6) << "Enable Autotuner " << m_name << std::endl;
        }
    }

void Autotuner::begin()
    {
    bool enabled = m_enabled;
    bool attached = m_attached;

    if (attached)
        {
        // wait until we have a new parameter value
        std::unique_lock<std::mutex> lk(m_mutex);
        m_cv.wait(lk, [=]{return m_have_param;});
        if (m_attached)
            {
            m_have_param = false;
            }

        m_cur_params_on_device = false;
        }

    #ifdef ENABLE_HIP
    if (!m_cur_params_on_device)
        {
        // sync up all GPUs so we can overwrite the parameters
        m_exec_conf->multiGPUBarrier();
        hipMemcpy(d_params, &m_current_param.front(),
            sizeof(unsigned int)*m_parameters.size(),
            hipMemcpyHostToDevice);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_cur_params_on_device = true;
        }
    #endif

    // skip if disabled
    if (!enabled && !attached)
        return;

    #ifdef ENABLE_HIP
    // if we are scanning, record a cuda event - otherwise do nothing
    if (attached || m_state == STARTUP || m_state == SCANNING)
        {
        m_cur_params_on_device = false;
        hipEventRecord(m_start, 0);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    #endif
    }

void Autotuner::end()
    {
    float sample = 0.0f;
    // handle timing updates if scanning or attached
    bool enabled = m_enabled;
    bool attached = m_attached;

    #ifdef ENABLE_HIP
    // skip if disabled
    if ((!enabled && !attached) || m_have_param)
        return;

    if (attached || m_state == STARTUP || m_state == SCANNING)
        {
        hipEventRecord(m_stop, 0);
        hipEventSynchronize(m_stop);
        hipEventElapsedTime(&sample, m_start, m_stop);

        if (! attached)
            {
            if (m_samples.find(m_current_element) == m_samples.end())
                {
                // initialize the storage for this element
                m_samples[m_current_element].resize(m_nsamples);
                }

            for (unsigned int i = 0; i < m_current_element.size(); ++i)
                m_samples[m_current_element][m_current_sample] = sample;
            }

        // print stats
        std::ostringstream oss;
        oss << "(";
        for (auto p: m_current_param)
            oss << p << ", ";
        oss << ")";

        m_exec_conf->msg->notice(9) << "Autotuner " << m_name << ": t[" << oss.str() << "," << m_current_sample
                                     << "] = " << sample << endl;

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    #endif

    if (attached)
        {
        m_last_sample = sample;

            {
            std::lock_guard<std::mutex> lk(m_mutex);

            // let tuner know we have a sample
            m_have_timing = true;
            }

        // signal that a new measurement is available
        m_cv.notify_one();
        }
    else
        {
        // handle state data updates and transitions
        if (m_state == STARTUP)
            {
            // move on to the next sample
            m_current_sample++;

            // if we hit the end of the samples, reset and move on to the next element
            if (m_current_sample >= m_nsamples)
                {
                m_current_sample = 0;

                bool done = false;
                for (unsigned int i = 0; i < m_current_element.size(); ++i)
                    {
                    if (m_enable_dim[i])
                        m_current_element[i]++;
                    else
                        continue;

                    if (m_current_element[i] >= m_parameters[i].size())
                        {
                        // find the next active tuning dimension:
                        unsigned int next_dim;
                        for (next_dim = i + 1; next_dim < m_current_element.size(); ++next_dim)
                            {
                            if (m_enable_dim[next_dim])
                                break;
                            }

                        // carry over
                        if (next_dim < m_current_element.size())
                            {
                            m_current_element[next_dim]++;
                            m_current_element[i] = 0;
                            }
                        else
                           {
                           done = true;
                           break;
                           }
                        }
                    }

                if (done)
                    {
                    for (unsigned int i = 0; i < m_current_element.size(); ++i)
                        {
                        if (m_enable_dim[i])
                            m_current_element[i] = 0;
                        }

                    m_state = IDLE;
                    m_current_param = computeOptimalParameter();
                    }
                else
                    {
                    // if moving on to the next element, update the cached parameter to set
                    for (unsigned int i = 0; i < m_parameters.size(); ++i)
                        m_current_param[i] = m_parameters[i][m_current_element[i]];
                    }
                }
            }
        else if (m_state == SCANNING)
            {
            bool done = false;
            for (unsigned int i = 0; i < m_current_element.size(); ++i)
                {
                if (m_enable_dim[i])
                    m_current_element[i]++;
                else
                    continue;

                if (m_current_element[i] >= m_parameters[i].size())
                    {
                    // find the next active tuning dimension:
                    unsigned int next_dim;
                    for (next_dim = i + 1; next_dim < m_current_element.size(); ++next_dim)
                        {
                        if (m_enable_dim[next_dim])
                            break;
                        }

                    // carry over
                    if (next_dim < m_current_element.size())
                        {
                        m_current_element[next_dim]++;
                        m_current_element[i] = 0;
                        }
                    else
                       {
                       done = true;
                       break;
                       }
                    }
                }

            if (done)
                {
                for (unsigned int i = 0; i < m_current_element.size(); ++i)
                    {
                    if (m_enable_dim[i])
                        m_current_element[i] = 0;
                    }

                m_state = IDLE;
                m_current_param = computeOptimalParameter();
                m_current_sample = (m_current_sample + 1) % m_nsamples;
                }
            else
                {
                // if moving on to the next element, update the cached parameter to set
                for (unsigned int i = 0; i < m_parameters.size(); ++i)
                    m_current_param[i] = m_parameters[i][m_current_element[i]];
                }
            }
        else if (m_state == IDLE)
            {
            // increment the calls counter and see if we should transition to the scanning state
            m_calls++;

            if (m_calls > m_period)
                {
                // reset state for the next time
                m_calls = 0;

                // initialize a scan
                for (unsigned int i = 0; i < m_parameters.size(); ++i)
                    m_current_param[i] = m_parameters[i][m_current_element[i]];
                m_state = SCANNING;
                m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " - beginning scan" << std::endl;
                }
            }
        }
    }

/*! \returns The optimal parameter given the current data in m_samples

    computeOptimalParameter computes the median time among all samples for a given element. It then chooses the
    fastest time (with the lowest index breaking a tie) and returns the parameter that resulted in that time.
*/
std::vector<unsigned int> Autotuner::computeOptimalParameter()
    {
    bool is_root = true;

    #ifdef ENABLE_MPI
    unsigned int nranks = 0;
    if (m_sync)
        {
        nranks = m_exec_conf->getNRanks();
        is_root = !m_exec_conf->getRank();
        }
    #endif

    // start by computing the median for each element
    std::vector<float> v;
    for (auto elem: m_samples)
        {
        v = elem.second;
        #ifdef ENABLE_MPI
        if (m_sync && nranks)
            {
            // combine samples from all ranks on rank zero
            std::vector< std::vector<float> > all_v;
            MPI_Barrier(m_exec_conf->getMPICommunicator());
            gather_v(v, all_v, 0, m_exec_conf->getMPICommunicator());
            if (is_root)
                {
                v.clear();
                assert(all_v.size() == nranks);
                for (unsigned int j = 0; j < nranks; ++j)
                    v.insert(v.end(), all_v[j].begin(), all_v[j].end());
                }
            }
        #endif
        if (is_root)
            {
            if (m_mode == mode_avg)
                {
                // compute average
                float sum = 0.0f;
                for (std::vector<float>::iterator it = v.begin(); it != v.end(); ++it)
                    sum += *it;
                m_sample_median[elem.first] = sum/v.size();
                }
            else
                {
                // compute median
                size_t n = v.size() / 2;
                nth_element(v.begin(), v.begin()+n, v.end());
                m_sample_median[elem.first] = v[n];
                }
            }
        }

    std::vector<unsigned int> opt(m_parameters.size());

    if (is_root)
        {
        // now find the minimum times in the medians
        float min = m_sample_median.begin()->second;
        std::vector< unsigned int > min_idx = m_sample_median.begin()->first;

        for (auto elem: m_sample_median)
            {
            if (elem.second < min)
                {
                min = elem.second;
                min_idx = elem.first;
                }
            }

        // get the optimal param
        for (unsigned int i = 0; i < opt.size(); ++i)
            {
            assert(min_idx.size() == m_parameters.size());
            opt[i] = m_parameters[i][min_idx[i]];
            }

        // print stats
        std::ostringstream oss;
        oss << "(";
        for (auto p: opt)
            oss << p << ", ";
        oss << ")";

        m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " found optimal parameter " << oss.str() << endl;
        }

    #ifdef ENABLE_MPI
    if (m_sync && nranks) bcast(opt, 0, m_exec_conf->getMPICommunicator());
    #endif
    return opt;
    }

bool Autotuner::sanityCheck(const std::vector<unsigned int>& param)
    {
    // sanity check
    if (param.size() != m_parameters.size())
        {
        this->m_exec_conf->msg->warning() << "Dimensionality of tuning parameter is " << param.size() << ", expected "
            << m_parameters.size() << " for tuner " << m_name << std::endl;
        return false;
        }

    for (unsigned int i = 0; i < m_parameters.size(); ++i)
        {
        bool found = false;

        for (auto q: m_parameters[i])
            {
            if (q == param[i])
                found = true;
            }
        if (!found)
            {
            // print stats
            std::ostringstream oss;
            oss << "(";
            for (auto p: param)
                oss << p << ", ";
            oss << ")";

            this->m_exec_conf->msg->warning() << "Unknown tuning parameter " << oss.str() << " in dimension " << i << std::endl;
            return false;
            }
        }
    return true;
    }

//! Measure the execution time of the next kernel launch
/* \param param the launch parameter to be tested
 * \returns the execution time in ms
 *
 * This method is intended be called from a separate host thread and
 * only returns when the kernel launch has completed.
 */
float Autotuner::measure(const pybind11::list param)
    {
    if (!m_attached)
        {
        throw std::runtime_error("The Autotuner is not attached. Cannot measure kernel run time.\n");
        }

    std::vector<unsigned int> p;
    for (unsigned int i = 0; i < pybind11::len(param); ++i)
        p.push_back(pybind11::cast<unsigned int>(param[i]));

    if (!sanityCheck(p))
        {
        m_exec_conf->msg->warning() << "Returning timing of 0. Tuning may not be sucessful." << std::endl;
        return 0.0;
        }

        {
        std::lock_guard<std::mutex> lk(m_mutex);

        // set the new parameter
        m_current_param = p;
        m_have_param = true;
        }

    // signal that a new parameter is available, this unblocks a pending kernel launch
    m_cv.notify_one();

    float m;
        {
        std::unique_lock<std::mutex> lk(m_mutex);

        // wait until the result is available
        m_cv.wait(lk, [=]{return m_have_timing;});
        m = m_last_sample;
        m_have_timing = false;
        }

    return m;
    }

void Autotuner::setOptimalParameter(const pybind11::list opt)
    {
    if (! m_attached)
        {
        throw std::runtime_error("The Autotuner is not attached. Cannot set optimal parameter value.\n");
        }

    std::vector<unsigned int> p;
    for (unsigned int i = 0; i < pybind11::len(opt); ++i)
        p.push_back(pybind11::cast<unsigned int>(opt[i]));

    if (!sanityCheck(p))
        {
        m_exec_conf->msg->warning() << "Ignoring optimal parameter." << std::endl;
        return;
        }

        {
        std::lock_guard<std::mutex> lk(m_mutex);

        // set the new parameter
        m_current_param = p;
        m_have_param = true;

        // prevent automatic retuning
        m_attached = false;
        m_enabled = false;
        }

    // signal that a new parameter is available, this unblocks a pending kernel launch
    m_cv.notify_one();
    }

void export_Autotuner(py::module& m)
    {
    py::class_<Autotuner, std::shared_ptr<Autotuner> >(m,"Autotuner")
    .def(py::init< unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const std::string&,
        std::shared_ptr<ExecutionConfiguration> >())
    .def("getParam", &Autotuner::getParam)
    .def("setEnabled", &Autotuner::setEnabled)
    .def("isComplete", &Autotuner::isComplete)
    .def("setPeriod", &Autotuner::setPeriod)
    .def("getName", &Autotuner::getName)
    .def("getParameterList", &Autotuner::getParameterList)
    .def("getEnableDimension", &Autotuner::getEnableDimension)
    .def("attach", &Autotuner::attach, pybind11::call_guard<py::gil_scoped_release>())
    .def("measure", &Autotuner::measure, pybind11::call_guard<py::gil_scoped_release>())
    .def("setOptimalParameter", &Autotuner::setOptimalParameter, pybind11::call_guard<py::gil_scoped_release>())
    ;
    }
