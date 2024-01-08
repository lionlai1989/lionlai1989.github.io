---
layout: single
title: Raytracing in C++ and Multithreading
categories: [raytracing]
description: "."
toc: true
wip: false
date: 2023-12-15
---

https://juanchopanzacpp.wordpress.com/2013/02/26/concurrent-queue-c11/
https://www.reddit.com/r/GraphicsProgramming/comments/oyksf4/cpu_raytracer_multithreading/

Ray tracing, a powerful technique for rendering scenes, is ideal for showcasing the
prowess of multithreading in programming. In this post, I dive into the details of
converting a single-threaded ray tracing program into a multithreaded one using the
codebase from the
[Ray Tracing in One Weekend Book Series](https://github.com/RayTracing/raytracing.github.io),
specifically focusing on the second article, the
[Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html).

## The Fundamentals of Ray Tracing

Ray tracing involves simulating the trajectory of rays as they traverse a scene To keep
our focus on multithreading, I'll touch upon essential aspects of ray tracing related to
multithreading without delving into intricate ray tracing details.

In a nutshell, a primary ray emits from the observer's eye, intersects the midpoint of a
pixel, and collides with an object, resulting in an image formation. This process is
replicated for each pixel, ultimately crafting an image, such as that of a ball as shown
in the images below. The two images below are adapted from
[Scratchapixel](https://www.scratchapixel.com/index.html).

<div style="text-align: center; margin-bottom: 10px">
    <img src="/assets/images/2023-12-15/lightingnoshadow.gif"
      alt="windows tile raster"
      style="margin: auto; width: 100%; max-width: 400px; margin-bottom: 20px">
    <br>
    <i>A primary ray emits from the observer's eye and intersects the center of a pixel.</i>
</div>

<div style="text-align: center; margin-bottom: 10px">
    <img src="/assets/images/2023-12-15/pixelrender.gif"
      alt="windows tile raster"
      style="margin: auto; width: 100%; max-width: 400px; margin-bottom: 20px">
    <br>
    <i>Shooting multiple primary rays forms an image.</i>
</div>

It's important to note that I am grossly simplifying ray tracing in this post. Given the
post's primary focus on integrating multithreading into a single-threaded program,
intricate details of ray tracing are omitted. For an in-depth understanding of ray
tracing, I recommend consulting resources like
[the Ray Tracing in One Weekend Book Series](https://github.com/RayTracing/raytracing.github.io)
and
[an excellent tutorial on ray tracing from Scratchapixel](https://www.scratchapixel.com/index.html)..

### Anti-Aliasing

**Note:** Images in this section are adapted from
[OpenGL Anti Aliasing](https://learnopengl.com/Advanced-OpenGL/Anti-Aliasing).

Another crucial aspect of ray tracing involves anti-aliasing. Consider the following
example illustrating the concept:

<div style="display:flex; justify-content:center;">
    <div style="text-align: center; margin-bottom: 10px">
        <img src="/assets/images/2023-12-15/anti_aliasing_rasterization.png"
        alt="windows tile raster"
        style="margin: auto; width: 100%; max-width: 300px; margin-bottom: 20px">
        <br>
        <i>A rectangle is sampled with one middle point per pixel..</i>
    </div>
    <div style="text-align: center; margin-bottom: 10px">
        <img src="/assets/images/2023-12-15/anti_aliasing_rasterization_filled.png"
        alt="windows tile raster"
        style="margin: auto; width: 100%; max-width: 300px; margin-bottom: 20px">
        <br>
        <i>After sampling, the triangle is with non-smooth edges.</i>
    </div>
</div>

Choosing only a single sample at the center of a pixel yields an outcome resembling a
triangle with uneven edges, as depicted below. A more effective approach involves
selecting multiple sample points within each pixel and computing the pixel value by
averaging these samples. In the following example, four sample points are randomly
chosen from a pixel. The average of these four pixel values is then calculated and
assigned as the final value for the pixel.

<div style="display:flex; justify-content:center;">
    <div style="text-align: center; margin-bottom: 10px">
        <img src="/assets/images/2023-12-15/anti_aliasing_rasterization_samples.png"
        alt="windows tile raster"
        style="margin: auto; width: 100%; max-width: 300px; margin-bottom: 20px">
        <br>
        <i>Four sampling points are randomly picked within a pixel.</i>
    </div>
    <div style="text-align: center; margin-bottom: 10px">
        <img src="/assets/images/2023-12-15/anti_aliasing_rasterization_samples_filled.png"
        alt="windows tile raster"
        style="margin: auto; width: 100%; max-width: 300px; margin-bottom: 20px">
        <br>
        <i>The averaged pixel value better captures the shape of a triangle.</i>
    </div>
</div>

The above description is implemented with
[the code snippet](https://github.com/RayTracing/raytracing.github.io/blob/7dd9a8099904f4508940b8fcb9781d079d1886d1/src/TheNextWeek/camera.h#L42):

```cpp
std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

for (int j = 0; j < image_height; ++j) {
    std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
        color pixel_color(0,0,0);
        for (int sample = 0; sample < samples_per_pixel; ++sample) {
            ray r = get_ray(i, j);
            pixel_color += ray_color(r, max_depth, world);
        }
        write_color(std::cout, pixel_color, samples_per_pixel);
    }
}
```

An additional note is that the single-threaded program writes pixel values as integers
to `std::cout` and utilizes the Linux right triangle bracket (>) to append the numbers
to a file, for example, `build/theNextWeek > test.ppm`. This approach implies that
pixels must be sequentially written out. Consequently, this design choice forces the
sequential calculation and output of pixel values one after another.

## Transitioning from Single-Threaded to Multithreaded

After laying the groundwork for multithreading in ray tracing, let's explore the
practical implementation. There are three feasible approaches to introducing
multithreading into the ray tracing program:

-   Each newly spawned thread processes multiple samples for a pixel.
-   Each newly spawned thread processes all samples for a pixel.
-   Each newly spawned thread processes all pixels for a row.

The implementation detail of the above three is trivial. Thus, I will just explain how a
thread processes all pixels for a row. Essentially, a thread processes all pixels in a
row and stores the result into a vector of pixel of color. In the main thread, it will
wait for each thread finishing processing the whole row and get the returned vector of
pixel's color from `future` object.

my approach is to use multiple threads to process samples for each pixel. After a thread
is finished, it returns pixel value in this thread. The main thread will accumulate
pixels derived from each thread and write to standard output. Here is the code snippet:

```cpp
for (int j = 0; j < image_height; ++j) {
    // Multi thread
    std::shared_future<std::vector<color>> f = futures.front();
    auto vec_pixel_color = f.get();
    futures.pop();
    for (int i = 0; i < vec_pixel_color.size(); ++i) {
        write_color(std::cout, vec_pixel_color[i], samples_per_pixel);
    }
}
```

### Producer Thread and BlockingQueue

On my laptop, it has 16 CPUs. It means that if running a CPU-bound program, such as
raytracing as I did here, it doesn't make too much sense to spawn more than 16 threads
for each time instance. Because if running more than 16 threads, there are not enough
CPUs to run it. Bear this in mind, in order To achieve the above feature, we need a
producer thread and a blocking queue. A blocking queue is a thread-safe queue data
structure which guarantee the pushing and poping things into and from this pool is
thread-safe. The producer thread is responsible for pushing things into the blocking
queue if the thread pool is not full. Thus, the overall multithreading structure can be
described as below:

-   `BlockingQueue`: It is a queue-like data structure which supports thread-safe
    pushing and popping. In `push`, the condition variable `_cond` will wait until the
    queue is not full. while waiting, it atomically unlocks lock, blocks the current
    executing thread.

```cpp
template <typename T>
class BlockingQueue {
  private:
    std::mutex _mtx;
    std::condition_variable _cond;
    int _max_size;
    std::queue<T> _queue;

  public:
    BlockingQueue(int max_size) : _max_size(max_size) {
    }

    void push(T t) {
        std::unique_lock<std::mutex> lock(_mtx);
        _cond.wait(lock, [this]() { return _queue.size() < _max_size; });
        _queue.push(t);
        lock.unlock();
        _cond.notify_one();
    }

    T front() {
        std::unique_lock<std::mutex> lock(_mtx);
        _cond.wait(lock, [this]() { return !_queue.empty(); });
        return _queue.front();
    }

    void pop() {
        std::unique_lock<std::mutex> lock(_mtx);
        _cond.wait(lock, [this]() { return !_queue.empty(); });
        _queue.pop();
        lock.unlock();
        _cond.notify_one();
    }

    int size() {
        std::lock_guard<std::mutex> lock(_mtx);
        return _queue.size();
    }
};
```

-   `producer_func`, `lambda_func` and `producer_thread`: The producer function
    `producer_func` spawns new thread which processes pixels in a row and pushes the
    future into the `BlockingQueue`. One thing to note here is that if the size of
    `BlockingQueue` is 4, intuitively it seems there is maximum 4 threads running
    concurrently. But there are actually at most 5 threads running at any given time.
    Because while the queue is full of 4 futures, the fifth thread is spawned and run.
    It's just the future of the fifth thread is waiting to be pushed into the
    `BlockingQueue`. The lambda function `lambda_func` processes every pixels in a row
    and store the pixel value into a vector of color `vector<color>`. The
    `producer_thread` is a thread running `producer_func`.

```cpp
void producer_func(BlockingQueue<std::shared_future<std::vector<color>>> &futures, int image_height, int image_width, std::function<std::vector<color>(int, int)> lambda_func) {
    for (int j = 0; j < image_height; ++j) {
        // A new thread is spawned and pushed into BlockingQueue.
        // If BlockingQueue is full, this thread will unlock the lock and wait until an element is pop out.
        std::shared_future<std::vector<color>> f = std::async(std::launch::async, lambda_func, j, image_width);
        futures.push(f);
    }
}

auto lambda_func = [this, &world](int j, int width) {
    std::vector<color> vec_pixel_color(width, color(0, 0, 0));
    for (int i = 0; i < width; ++i) {
        color pixel_color(0,0,0);
        for (int sample = 0; sample < this->samples_per_pixel; ++sample) {
            const ray r = get_ray(i, j);
            pixel_color += ray_color(r, this->max_depth, world);
        }
        vec_pixel_color[i] = pixel_color;
    }
    return vec_pixel_color;
};

std::thread producer_thread(producer_func, std::ref(futures), image_height, image_width, lambda_func);
```

Putting things together, we can get:

```cpp
std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

const int CONCURRENCY = 4; // `std::thread::hardware_concurrency();` returns 16 on my laptop.
std::clog << "\rCONCURRENCY: " << CONCURRENCY << std::endl;
BlockingQueue<std::shared_future<std::vector<color>>> futures(CONCURRENCY); // CONCURRENCY = BlockingQueue's size

// The lambda function on which every spawned thread is going to run. Every
// thread process the whole given j-th row.
auto lambda_func = [this, &world](int j, int width) {
    std::vector<color> vec_pixel_color(width, color(0, 0, 0));
    for (int i = 0; i < width; ++i) {
        color pixel_color(0,0,0);
        for (int sample = 0; sample < this->samples_per_pixel; ++sample) {
            const ray r = get_ray(i, j);
            pixel_color += ray_color(r, this->max_depth, world);
        }
        vec_pixel_color[i] = pixel_color;
    }
    return vec_pixel_color;
};
std::thread producer_thread(producer_func, std::ref(futures), image_height, image_width, lambda_func);

for (int j = 0; j < image_height; ++j) {
    std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;

    // Multi thread
    std::shared_future<std::vector<color>> f = futures.front();
    auto vec_pixel_color = f.get();
    futures.pop();
    for (int i = 0; i < vec_pixel_color.size(); ++i) {
        write_color(std::cout, vec_pixel_color[i], samples_per_pixel);
    }
}
producer_thread.join();
```

### Profile single threaded and multithreaded program

The ultimate way to verify a multithreaded program is to profile it and compare the
performance is linearly improved. After defining all compoments, I can compare the
multithreaded program with the baseline (origianl code). The image size is 1024 x 1024.
I choose 256 and 1024 samples per pixels. The baseline is defined as the original code
without using multithreading.

| Samples | Baseline | Concurrency=2 | Concurrency=4 |
| ------- | -------- | ------------- | ------------- |
| 256     |          | (sec)         | (sec)         |
| 1024    |          | (sec)         | (sec)         |

The result shows that using multithreading can reduce the running time. However, the
performace is not linearly improved with respect to the size of the `BlockingQueue`. I'd
expect that the runnign time would be quatered of the baseline when `Concurrency=4`.

There are few possibilities of this phenomenon. I listed them below:

-   This main program (main thread) writes to standard output sequentially pixel by
    pixel. It could be the reason that is blocking the main program popping futures out
    of the `BlockingQueue`. I tried disabling the writing pixel value to standard
    output, but there is not significant performance improvement.

-   Due to the nature of raytracing, it should be a great example of showcasing the
    power of multithreading because every pixel can be calculated independently and not
    related to the neighboring pixels. The only thing is shared among multiple threads
    in this program is `world`. `world` is an object which describes the scene. `world`
    is a constant reference `const hittable& world` that is read-only. My understanding
    of multithreading is that if a pass-by-reference shared object in different threads
    can only be read but not write, then mutex and lock are not needed when accessing
    them. (However, I wonder how it works in low level. How can two threads read the
    same memory address (pass-by-reference) without stepping on others' toe).
