/**
 * @file event.hpp
 *
 * @brief A CUDA event wrapper class and some associated
 * free-standing functions.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_EVENT_HPP_
#define CUDA_API_WRAPPERS_EVENT_HPP_

#include <cuda/api/types.hpp>

#include <cuda_runtime_api.h>

#include <chrono> // for duration types
#include <cuda/api/constants.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/ipc.hpp>

namespace cuda {

///@cond
class device_t;
class stream_t;
///@endcond

namespace event {

namespace detail_ {

/**
 * Schedule a specified event to occur (= to fire) when all activities
 * already scheduled on the stream have concluded.
 *
 * @param stream_handle handle of the stream (=queue) where to enqueue the event occurrence
 * @param event_handle Event to be made to occur on stream @ref stream_handle
 */
inline void enqueue(stream::handle_t stream_handle, handle_t event_handle) {
	auto status = cuEventRecord(event_handle, stream_handle);
	cuda::throw_if_error(status,
		"Failed recording " + event::detail_::identify(event_handle)
		+ " on " + stream::detail_::identify(stream_handle));
}

constexpr unsigned inline make_flags(bool uses_blocking_sync, bool records_timing, bool interprocess)
{
	return
		  ( uses_blocking_sync  ? CU_EVENT_BLOCKING_SYNC : 0  )
		| ( records_timing      ? 0 : CU_EVENT_DISABLE_TIMING )
		| ( interprocess        ? CU_EVENT_INTERPROCESS : 0  );
}

} // namespace detail_

} // namespace event

///@cond
class event_t;
///@endcond

namespace event {

/**
 * @brief Wrap an existing CUDA event in a @ref event_t instance.
 *
 * @note This is a named constructor idiom, existing of direct access to the ctor
 * of the same signature, to emphasize that a new event is _not_ created.
 *
 * @param context_handle Handle of the context in which this event was created
 * @param event_handle handle of the pre-existing event
 * @param take_ownership When set to `false`, the CUDA event
 * will not be destroyed along with proxy; use this setting
 * when temporarily working with a stream existing irrespective of
 * the current context and outlasting it. When set to `true`,
 * the proxy class will act as it does usually, destroying the event
 * when being destructed itself.
 * @return an event wrapper associated with the specified event
 */
event_t wrap(
	device::id_t       device_id,
	context::handle_t  context_handle,
	handle_t           event_handle,
	bool               take_ownership = false) noexcept;

::std::string identify(const event_t& event);

} // namespace event

inline void synchronize(const event_t& event);

/**
 * @brief Wrapper class for a CUDA event
 *
 * Use this class - built around an event handle - to perform almost, if not all,
 * event-related operations the CUDA Runtime API is capable of.
 *
 * @note By default this class has RAII semantics, i.e. it has the runtime create
 * an event on construction and destroy it on destruction, and isn't merely
 * an ephemeral wrapper one could apply and discard; but this second kind of
 * semantics is also (sort of) supported, through the @ref event_t::owning field.
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::device_t and @ref cuda::stream_t
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the event is a const-respecting operation on this class.
 */
class event_t {

public: // data member non-mutator getters
	/// The raw CUDA ID for the device w.r.t. which the event is defined
	device::id_t      device_id()       const noexcept { return device_id_; };

	/// The raw CUDA handle for the context in which the represented stream is defined.
	context::handle_t context_handle()  const noexcept { return context_handle_; }

	/// The raw CUDA handle for this event
	event::handle_t   handle()          const noexcept { return handle_; }

	/// True if this wrapper is responsible for telling CUDA to destroy the event upon the wrapper's own destruction
	bool              is_owning()       const noexcept { return owning; }

	/// The device w.r.t. which the event is defined
	device_t          device()          const;

	/// The context in which this stream was defined.
	context_t         context()         const;



public: // other non-mutator methods

	/**
	 * Has this event already occurred, or is it still pending on a stream?
	 *
	 * @note an event can occur multiple times, but in the context of this
	 * method, it only has two states: pending (on a stream) and has_occured.
	 *
	 * @return if all work on the stream with which the event was recorded
	 * has been completed, returns true; if there is pending work on that stream
	 * before the point of recording, returns false; if the event has not
	 * been recorded at all, returns true.
	 */
	bool has_occurred() const
	{
		auto status = cuEventQuery(handle_);
		if (status == cuda::status::success) return true;
		if (status == cuda::status::not_ready) return false;
		throw cuda::runtime_error(status,
			"Could not determine whether " + event::detail_::identify(handle_)
			+ "has already occurred or not.");
	}

	/**
	 * An alias for {@ref event_t::has_occurred()} - to conform to how the CUDA runtime
	 * API names this functionality
	 */
	bool query() const { return has_occurred(); }


public: // other mutator methods

	/**
	 * Schedule a specified event to occur (= to fire) when all activities
	 * already scheduled on the event's device's default stream have concluded.
	 *
	 * @note No protection against repeated calls.
	 */
	void record() const
	{
		event::detail_::enqueue(stream::default_stream_handle, handle_);
	}

	/**
	 * Schedule a specified event to occur (= to fire) when all activities
	 * already scheduled on the stream have concluded.
	 *
	 * @note No protection against repeated calls.
	 */
	void record(const stream_t& stream) const;

	/**
	 * Records the event and ensures it has occurred before returning
	 * (by synchronizing the stream).
	 *
	 * @note No protection against repeated calls.
	 */
	void fire(const stream_t& stream) const;

	/**
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after this event has occurred (see @ref has_occurred() ).
	 */
	void synchronize() const
	{
		return cuda::synchronize(*this);
	}

protected: // constructors

	event_t(device::id_t device_id, context::handle_t context_handle, event::handle_t event_handle, bool take_ownership) noexcept
	: device_id_(device_id), context_handle_(context_handle), handle_(event_handle), owning(take_ownership) { }

public: // friendship

	friend event_t event::wrap(device::id_t, context::handle_t context_handle, event::handle_t event_handle, bool take_ownership) noexcept;

public: // constructors and destructor

	event_t(const event_t& other) noexcept : event_t(other.device_id_, other.context_handle_, other.handle_, false) { }

	event_t(event_t&& other) noexcept :
		event_t(other.device_id_, other.context_handle_, other.handle_, other.owning)
	{
		other.owning = false;
	};

	~event_t()
	{
		if (owning) {
			cuEventDestroy(handle_);
				// Note: "Swallowing" any potential error to avoid std::terminate(); also,
				// because the context cannot possibly exist after this call.
		}
	}

public: // operators

	event_t& operator=(const event_t& other) = delete;
	event_t& operator=(event_t&& other) = delete;

protected: // data members
	const device::id_t       device_id_;
	const context::handle_t  context_handle_;
	const event::handle_t    handle_;
	bool                     owning;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
};

namespace event {

/**
 * @brief The type used by the CUDA Runtime API to represent the time difference
 * between pairs of events.
 */
using duration_t = ::std::chrono::duration<float, ::std::milli>;

/**
 * Determine (inaccurately) the elapsed time between two events
 *
 * @note  Q: Why the weird output type?
 *        A: This is what the CUDA Runtime API itself returns
 *
 * @param start first timepoint event
 * @param end second, later, timepoint event
 * @return the difference in the (inaccurately) measured time, in msec
 */
inline duration_t time_elapsed_between(const event_t& start, const event_t& end)
{
	float elapsed_milliseconds;
	auto status = cuEventElapsedTime(&elapsed_milliseconds, start.handle(), end.handle());
	cuda::throw_if_error(status, "determining the time elapsed between events");
	return duration_t { elapsed_milliseconds };
}

inline event_t wrap(
	device::id_t       device_id,
	context::handle_t  context_handle,
	handle_t           event_handle,
	bool               take_ownership) noexcept
{
	return { device_id, context_handle, event_handle, take_ownership };
}

namespace detail_ {

inline ::std::string identify(const event_t& event)
{
	return identify(event.handle(), event.context_handle(), event.device_id());
}

// Note: For now, event_t's need their device's ID - even if it's the current device;
// that explains the requirement in this function's interface
inline event_t create_in_current_context(
	device::id_t       current_device_id,
	context::handle_t  current_context_handle,
	bool               uses_blocking_sync,
	bool               records_timing,
	bool               interprocess)
{
	auto flags = make_flags(uses_blocking_sync, records_timing, interprocess);
	cuda::event::handle_t new_event_handle;
	auto status = cuEventCreate(&new_event_handle, flags);
	cuda::throw_if_error(status, "failed creating a CUDA event associated with the current device");
	// Note: We're trusting CUDA to actually have succeeded if it reports success,
	// so we're not checking the newly-created event handle - which is really just
	// a pointer - for nullness
	bool take_ownership = true;
	return wrap(current_device_id, current_context_handle, new_event_handle, take_ownership);
}

/**
 * @note see @ref cuda::event::create()
 */

inline event_t create(
	device::id_t       device_id,
	context::handle_t  context_handle,
	bool               uses_blocking_sync,
	bool               records_timing,
	bool               interprocess)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle);
	return detail_::create_in_current_context(device_id, context_handle, uses_blocking_sync, records_timing, interprocess);
}

} // namespace detail_

/**
 * @brief creates a new execution stream on a device.
 *
 * @param device              The device on which to create the new stream
 * @param uses_blocking_sync  When synchronizing on this new event, shall a thread busy-wait for it, or block?
 * @param records_timing      Can this event be used to record time values (e.g. duration between events)?
 * @param interprocess        Can multiple processes work with the constructed event?
 * @return The constructed event proxy
 *
 * @note Creating an event
 */
inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync = sync_by_busy_waiting, // Yes, that's the runtime default
	bool       records_timing     = do_record_timings,
	bool       interprocess       = not_interprocess);

} // namespace event

/**
 * Waits for a specified event to conclude before returning control
 * to the calling code.
 *
 * @todo Determine how this waiting takes place (as opposed to stream
 * synchronization).
 *
 * @param event the event for whose occurrence to wait; must be scheduled
 * to occur on some stream (possibly the different stream)
 */
inline void synchronize(const event_t& event)
{
	auto context_handle = event.context_handle();
	auto event_handle = event.handle();
	context::current::detail_::scoped_override_t context_for_this_scope(context_handle);
	auto status = cuEventSynchronize(event_handle);
	throw_if_error(status, "Failed synchronizing " + event::detail_::identify(event));
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_EVENT_HPP_
