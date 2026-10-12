#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct
{
    float mass;
    float yaw_inertia;
    float front_axle;
    float rear_axle;
    float gravity;
    float grip;
    float front_stiffness;
    float rear_stiffness;
    float engine_force;
    float brake_force;
    float drag;
    float rolling_drag;
    float steering_limit;
    float steering_rate;
    float velocity_min;
    float velocity_max;
    float lateral_limit;
    float yaw_limit;
    float minimum_effective_speed;
    float low_speed_damping;
    float straight_epsilon;
    float speed_normalization;
    float lateral_normalization;
    float yaw_normalization;
    float car_length;
    float car_width;
} VehicleParams;

#if defined(_WIN32)
#define RACING_API __declspec(dllexport)
#else
#define RACING_API
#endif

typedef struct
{
    float *control_points;
    int32_t point_count;
    float road_width;
    float *lookahead_distances;
    int32_t lookahead_count;
    int32_t smoothing;
    int32_t samples_per_span;
    float cell_size;

    float step_dt;
    int32_t substeps;
    VehicleParams vehicle;
    int32_t max_steps;
} RacingConfig;

typedef struct
{
    float x, y, heading;
    float forward_speed, lateral_speed, yaw_rate;
    float steering_angle;
    float last_s, progress;
    int32_t steps;
    float next_lap_progress;
    int32_t last_lap_step;
    float front_slip, rear_slip;
} CarState;

typedef struct
{
    float ax, ay, dx, dy;
    float length, start_s;
} RoadSegment;

typedef struct
{
    int32_t segment, row;
    float progress, best_distance2;
} GridCell;

typedef struct
{
    RoadSegment *segments;
    float *normals;
    int32_t segment_count;
    float lap_length;

    GridCell *cells;
    int32_t grid_width, grid_height;
    float grid_x, grid_y, cell_size;

    float *lookahead;
    int32_t on_road_count;
} CompiledMap;

typedef struct
{
    RacingConfig config;
    int32_t num_envs;
    CarState *cars;
    CompiledMap map;
    uint64_t rng;
    float *road_scales;
    float speed_scale, lateral_scale, yaw_scale, steering_scale;
} RacingEnv;

static float clampf(float value, float low, float high)
{
    return fminf(fmaxf(value, low), high);
}

typedef struct CarPhysicsState
{
    float x;
    float y;
    float heading;
    float heading_cos;
    float heading_sin;
    float vx;
    float vy;
    float yaw_rate;
    float steer;
    float front_slip;
    float rear_slip;
} CarPhysicsState;

typedef struct CarControl
{
    float steer_action;
    float throttle;
    float brake;
} CarControl;

static inline float clamp_f(float value, float lo, float hi)
{
    return value < lo ? lo : (value > hi ? hi : value);
}

static inline float wrap_pi(float angle)
{
    const float pi = 3.14159265358979323846f;
    const float two_pi = 6.28318530717958647692f;
    while (angle > pi)
    {
        angle -= two_pi;
    }
    while (angle < -pi)
    {
        angle += two_pi;
    }
    return angle;
}

static inline void car_physics_substep(CarPhysicsState *car, const CarControl *control, float dt, const VehicleParams *p)
{
    float rear_limit = p->grip * (p->mass * p->gravity * p->front_axle / (p->front_axle + p->rear_axle));
    float front_limit = p->grip * (p->mass * p->gravity * p->rear_axle / (p->front_axle + p->rear_axle));
    float steer_target = control->steer_action * p->steering_limit;
    float steer_delta = steer_target - car->steer;
    float max_steer_delta = p->steering_rate * dt;
    steer_delta = clamp_f(steer_delta, -max_steer_delta, max_steer_delta);
    car->steer += steer_delta;

    float effective_vx = car->vx;
    if (effective_vx >= 0.0f && effective_vx < p->minimum_effective_speed)
    {
        effective_vx = p->minimum_effective_speed;
    }
    else if (effective_vx < 0.0f && effective_vx > -p->minimum_effective_speed)
    {
        effective_vx = -p->minimum_effective_speed;
    }

    float front_slip = atan2f(car->vy + p->front_axle * car->yaw_rate, fabsf(effective_vx)) - car->steer;
    float rear_slip = atan2f(car->vy - p->rear_axle * car->yaw_rate, fabsf(effective_vx));

    float front_lat = -p->front_stiffness * front_slip;
    float rear_lat = -p->rear_stiffness * rear_slip;
    float previous_vx = car->vx;
    float speed_sign = car->vx > 0.0f ? 1.0f : (car->vx < 0.0f ? -1.0f : 0.0f);
    float rear_long = control->throttle * p->engine_force - control->brake * p->brake_force * speed_sign;
    float drag = p->drag * car->vx * fabsf(car->vx);
    float rolling = p->rolling_drag * speed_sign;

    front_lat = clamp_f(front_lat, -front_limit, front_limit);
    rear_long = clamp_f(rear_long, -rear_limit, rear_limit);

    float rear_lat_limit_sq = rear_limit * rear_limit - rear_long * rear_long;
    float rear_lat_limit = rear_lat_limit_sq > 0.0f ? sqrtf(rear_lat_limit_sq) : 0.0f;
    rear_lat = clamp_f(rear_lat, -rear_lat_limit, rear_lat_limit);

    float steer_sin = sinf(car->steer);
    float steer_cos = cosf(car->steer);
    float ax = (rear_long - drag - rolling - front_lat * steer_sin) / p->mass + car->vy * car->yaw_rate;
    float ay = (rear_lat + front_lat * steer_cos) / p->mass - car->vx * car->yaw_rate;
    float yaw_accel = (p->front_axle * front_lat * steer_cos - p->rear_axle * rear_lat) / p->yaw_inertia;

    car->vx += ax * dt;
    car->vy += ay * dt;
    car->yaw_rate += yaw_accel * dt;

    car->vx = clamp_f(car->vx, p->velocity_min, p->velocity_max);
    if (control->throttle <= 0.0f &&
        ((previous_vx > 0.0f && car->vx <= 0.0f) || (previous_vx < 0.0f && car->vx >= 0.0f) ||
         (previous_vx == 0.0f && car->vx < 0.0f) || fabsf(car->vx) < 0.0001f))
    {
        car->vx = 0.0f;
    }
    car->vy = clamp_f(car->vy, -p->lateral_limit, p->lateral_limit);
    car->yaw_rate = clamp_f(car->yaw_rate, -p->yaw_limit, p->yaw_limit);

    if (fabsf(car->vx) < p->minimum_effective_speed)
    {
        car->vy *= p->low_speed_damping;
        car->yaw_rate *= p->low_speed_damping;
    }

    car->heading += car->yaw_rate * dt;
    car->heading = wrap_pi(car->heading);
    car->heading_cos = cosf(car->heading);
    car->heading_sin = sinf(car->heading);

    car->x += (car->heading_cos * car->vx - car->heading_sin * car->vy) * dt;
    car->y += (car->heading_sin * car->vx + car->heading_cos * car->vy) * dt;

    car->front_slip = front_slip;
    car->rear_slip = rear_slip;
}

static void advance_car(CarState *car, const RacingConfig *config, float steering, float pedal, float dt)
{
    CarPhysicsState state = {
        car->x, car->y, car->heading, cosf(car->heading), sinf(car->heading),
        car->forward_speed, car->lateral_speed, car->yaw_rate, car->steering_angle,
        car->front_slip, car->rear_slip};
    steering = isfinite(steering) ? steering : 0.0f;
    pedal = isfinite(pedal) ? pedal : 0.0f;
    CarControl control;
    control.steer_action = clamp_f(steering, -1.0f, 1.0f);
    pedal = clamp_f(pedal, -1.0f, 1.0f);
    control.throttle = pedal > 0.0f ? pedal : 0.0f;
    control.brake = pedal < 0.0f ? -pedal : 0.0f;
    int steps = (int)ceilf(dt / (1.0f / 60.0f) - 1e-6f);
    for (int i = 0; i < steps; i++)
        car_physics_substep(&state, &control, dt / steps, &config->vehicle);
    car->x = state.x;
    car->y = state.y;
    car->heading = state.heading;
    car->forward_speed = state.vx;
    car->lateral_speed = state.vy;
    car->yaw_rate = state.yaw_rate;
    car->steering_angle = state.steer;
    car->front_slip = state.front_slip;
    car->rear_slip = state.rear_slip;
}

static float catmull(float p0, float p1, float p2, float p3, float t)
{
    return 0.5f * (2.0f * p1 + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t * t + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t * t * t);
}

static float *sample_centerline(const RacingConfig *config, int32_t *count)
{
    assert(config->control_points && config->point_count >= 3);
    assert(config->smoothing == 0 || config->smoothing == 1);
    assert(config->samples_per_span > 0);

    int32_t samples = config->smoothing ? config->samples_per_span : 1;
    assert(config->point_count <= INT32_MAX / samples);
    *count = config->point_count * samples;
    float *points = malloc((size_t)*count * 2 * sizeof(float));
    assert(points);

    if (!config->smoothing)
    {
        memcpy(points, config->control_points, (size_t)*count * 2 * sizeof(float));
        return points;
    }

    int32_t n = config->point_count;
    for (int32_t i = 0; i < n; i++)
    {
        int32_t p0 = (i + n - 1) % n;
        int32_t p1 = i;
        int32_t p2 = (i + 1) % n;
        int32_t p3 = (i + 2) % n;
        for (int32_t j = 0; j < samples; j++)
        {
            float t = (float)j / samples;
            for (int32_t axis = 0; axis < 2; axis++)
            {
                points[2 * (i * samples + j) + axis] = catmull(
                    config->control_points[2 * p0 + axis],
                    config->control_points[2 * p1 + axis],
                    config->control_points[2 * p2 + axis],
                    config->control_points[2 * p3 + axis], t);
            }
        }
    }
    return points;
}

static void build_segments(CompiledMap *map, const float *points, int32_t count)
{
    map->segment_count = count;
    map->segments = malloc((size_t)count * sizeof(RoadSegment));
    map->normals = malloc((size_t)count * 2 * sizeof(float));
    assert(map->segments && map->normals);

    float distance = 0.0f;
    for (int32_t i = 0; i < count; i++)
    {
        int32_t next = (i + 1) % count;
        RoadSegment *segment = &map->segments[i];
        segment->ax = points[2 * i];
        segment->ay = points[2 * i + 1];
        segment->dx = points[2 * next] - segment->ax;
        segment->dy = points[2 * next + 1] - segment->ay;
        segment->length = hypotf(segment->dx, segment->dy);
        assert(isfinite(segment->length) && segment->length > 1e-6f);
        segment->start_s = distance;
        distance += segment->length;
    }
    map->lap_length = distance;

    for (int32_t i = 0; i < count; i++)
    {
        const RoadSegment *before = &map->segments[(i + count - 1) % count];
        const RoadSegment *after = &map->segments[i];
        float tx = before->dx / before->length + after->dx / after->length;
        float ty = before->dy / before->length + after->dy / after->length;
        float length = hypotf(tx, ty);
        if (length < 1e-6f)
        {
            tx = after->dx / after->length;
            ty = after->dy / after->length;
        }
        else
        {
            tx /= length;
            ty /= length;
        }
        map->normals[2 * i] = -ty;
        map->normals[2 * i + 1] = tx;
    }
}

static void build_grid(CompiledMap *map, const RacingConfig *config, const float *points)
{
    int32_t count = map->segment_count;
    float min_x = points[0], max_x = points[0];
    float min_y = points[1], max_y = points[1];
    for (int32_t i = 1; i < count; i++)
    {
        min_x = fminf(min_x, points[2 * i]);
        max_x = fmaxf(max_x, points[2 * i]);
        min_y = fminf(min_y, points[2 * i + 1]);
        max_y = fmaxf(max_y, points[2 * i + 1]);
    }

    float half_width = config->road_width * 0.5f;
    float padding = half_width + config->cell_size;
    map->cell_size = config->cell_size;
    map->grid_x = min_x - padding;
    map->grid_y = min_y - padding;
    map->grid_width = (int32_t)ceilf((max_x - min_x + 2.0f * padding) / map->cell_size);
    map->grid_height = (int32_t)ceilf((max_y - min_y + 2.0f * padding) / map->cell_size);
    assert(map->grid_width > 0 && map->grid_height > 0);

    size_t cell_count = (size_t)map->grid_width * map->grid_height;
    assert(cell_count <= INT32_MAX);
    map->cells = malloc(cell_count * sizeof(GridCell));
    assert(map->cells);
    for (size_t i = 0; i < cell_count; i++)
    {
        map->cells[i].segment = -1;
        map->cells[i].row = -1;
        map->cells[i].progress = 0.0f;
        map->cells[i].best_distance2 = FLT_MAX;
    }

    for (int32_t i = 0; i < count; i++)
    {
        const RoadSegment *segment = &map->segments[i];
        float bx = segment->ax + segment->dx;
        float by = segment->ay + segment->dy;
        int32_t x0 = (int32_t)floorf((fminf(segment->ax, bx) - half_width - map->grid_x) / map->cell_size);
        int32_t x1 = (int32_t)ceilf((fmaxf(segment->ax, bx) + half_width - map->grid_x) / map->cell_size);
        int32_t y0 = (int32_t)floorf((fminf(segment->ay, by) - half_width - map->grid_y) / map->cell_size);
        int32_t y1 = (int32_t)ceilf((fmaxf(segment->ay, by) + half_width - map->grid_y) / map->cell_size);
        if (x0 < 0)
            x0 = 0;
        if (y0 < 0)
            y0 = 0;
        if (x1 >= map->grid_width)
            x1 = map->grid_width - 1;
        if (y1 >= map->grid_height)
            y1 = map->grid_height - 1;

        for (int32_t y = y0; y <= y1; y++)
        {
            float qy = map->grid_y + (y + 0.5f) * map->cell_size;
            for (int32_t x = x0; x <= x1; x++)
            {
                float qx = map->grid_x + (x + 0.5f) * map->cell_size;
                float u = clampf(((qx - segment->ax) * segment->dx + (qy - segment->ay) * segment->dy) / (segment->length * segment->length), 0.0f, 1.0f);
                float dx = qx - (segment->ax + u * segment->dx);
                float dy = qy - (segment->ay + u * segment->dy);
                float distance2 = dx * dx + dy * dy;
                GridCell *cell = &map->cells[(size_t)y * map->grid_width + x];
                if (distance2 < cell->best_distance2)
                {
                    cell->best_distance2 = distance2;
                    cell->segment = i;
                    cell->progress = segment->start_s + u * segment->length;
                }
            }
        }
    }

    float road_radius2 = half_width * half_width;
    for (size_t i = 0; i < cell_count; i++)
    {
        if (map->cells[i].best_distance2 <= road_radius2)
            map->cells[i].row = map->on_road_count++;
    }
    assert(map->on_road_count > 0);
}

static const GridCell *road_cell(const CompiledMap *map, float x, float y)
{
    float gx = (x - map->grid_x) / map->cell_size;
    float gy = (y - map->grid_y) / map->cell_size;

    if (!(gx >= 0.0f && gx < map->grid_width &&
          gy >= 0.0f && gy < map->grid_height))
        return NULL;

    const GridCell *cell = &map->cells[(size_t)(int32_t)gy * map->grid_width + (int32_t)gx];
    return cell->row >= 0 ? cell : NULL;
}

static void road_edges_at(
    const CompiledMap *map, float progress, float road_width, float *edges)
{
    progress = fmodf(progress, map->lap_length);
    if (progress < 0.0f)
        progress += map->lap_length;

    int32_t low = 0, high = map->segment_count;
    while (low + 1 < high)
    {
        int32_t middle = low + (high - low) / 2;
        if (map->segments[middle].start_s <= progress)
            low = middle;
        else
            high = middle;
    }

    const RoadSegment *segment = &map->segments[low];
    float u = clampf((progress - segment->start_s) / segment->length, 0.0f, 1.0f);
    int32_t next = (low + 1) % map->segment_count;
    float nx = (1.0f - u) * map->normals[2 * low] + u * map->normals[2 * next];
    float ny = (1.0f - u) * map->normals[2 * low + 1] + u * map->normals[2 * next + 1];
    float normal_length = hypotf(nx, ny);
    if (normal_length < 1e-6f)
    {
        nx = -segment->dy / segment->length;
        ny = segment->dx / segment->length;
    }
    else
    {
        nx /= normal_length;
        ny /= normal_length;
    }

    float cx = segment->ax + u * segment->dx;
    float cy = segment->ay + u * segment->dy;
    float offset = road_width * 0.5f;
    edges[0] = cx + offset * nx;
    edges[1] = cy + offset * ny;
    edges[2] = cx - offset * nx;
    edges[3] = cy - offset * ny;
}

static void build_lookahead(CompiledMap *map, const RacingConfig *config)
{
    size_t values = (size_t)map->on_road_count * config->lookahead_count * 4;
    map->lookahead = malloc(values * sizeof(float));
    assert(map->lookahead);

    size_t cell_count = (size_t)map->grid_width * map->grid_height;
    for (size_t i = 0; i < cell_count; i++)
    {
        const GridCell *cell = &map->cells[i];
        if (cell->row < 0)
            continue;
        for (int32_t k = 0; k < config->lookahead_count; k++)
        {
            float *edges = &map->lookahead[((size_t)cell->row * config->lookahead_count + k) * 4];
            road_edges_at(map, cell->progress + config->lookahead_distances[k], config->road_width, edges);
        }
    }
}

static void compile_map(CompiledMap *map, const RacingConfig *config)
{
    assert(isfinite(config->road_width) && config->road_width > 0.0f);
    assert(isfinite(config->cell_size) && config->cell_size > 0.0f);
    assert(config->lookahead_distances && config->lookahead_count > 0);
    for (int32_t i = 0; i < config->lookahead_count; i++)
        assert(isfinite(config->lookahead_distances[i]));
    memset(map, 0, sizeof(*map));

    int32_t count;
    float *points = sample_centerline(config, &count);
    build_segments(map, points, count);
    build_grid(map, config, points);
    build_lookahead(map, config);
    free(points);
}

RACING_API RacingEnv *race_create(
    const RacingConfig *config,
    int32_t num_envs,
    uint64_t seed)
{
    assert(config && num_envs > 0);
    assert(config->control_points && config->point_count >= 3);
    assert(config->lookahead_distances && config->lookahead_count > 0);
    assert(isfinite(config->step_dt) && config->step_dt > 0.0f);
    assert(config->substeps > 0 && config->max_steps >= 0);

    RacingEnv *env = calloc(1, sizeof(RacingEnv));
    assert(env);
    env->config = *config;
    env->num_envs = num_envs;
    env->rng = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);

    size_t point_bytes = (size_t)config->point_count * 2 * sizeof(float);
    size_t lookahead_bytes = (size_t)config->lookahead_count * sizeof(float);
    env->config.control_points = malloc(point_bytes);
    env->config.lookahead_distances = malloc(lookahead_bytes);
    env->cars = calloc((size_t)num_envs, sizeof(CarState));
    assert(env->config.control_points && env->config.lookahead_distances && env->cars);
    memcpy(env->config.control_points, config->control_points, point_bytes);
    memcpy(env->config.lookahead_distances, config->lookahead_distances, lookahead_bytes);

    assert(config->road_width > 0.0f);
    const VehicleParams *p = &config->vehicle;
    assert(p->mass > 0.0f && p->yaw_inertia > 0.0f);
    assert(p->front_axle > 0.0f && p->rear_axle > 0.0f);
    assert(p->grip > 0 && p->engine_force > 0 && p->brake_force > 0);
    assert(p->steering_limit > 0 && p->steering_limit < 1.570796 && p->steering_rate > 0);
    assert(p->speed_normalization > 0.0f && p->lateral_normalization > 0.0f && p->yaw_normalization > 0.0f);

    env->road_scales = malloc((size_t)config->lookahead_count * 2 * sizeof(float));
    assert(env->road_scales);
    for (int32_t i = 0; i < config->lookahead_count; i++)
    {
        float d = config->lookahead_distances[i];
        assert(isfinite(d) && d >= 0.0f);
        env->road_scales[2 * i] = 1.0f / (2.0f * d + config->road_width);
        env->road_scales[2 * i + 1] = 1.0f / (d + config->road_width);
    }
    env->speed_scale = 1.0f / p->speed_normalization;
    env->lateral_scale = 1.0f / p->lateral_normalization;
    env->yaw_scale = 1.0f / p->yaw_normalization;
    env->steering_scale = 1.0f / p->steering_limit;

    compile_map(&env->map, &env->config);
    return env;
}

RACING_API void race_destroy(RacingEnv *env)
{
    if (!env)
        return;
    free(env->map.segments);
    free(env->map.normals);
    free(env->map.cells);
    free(env->map.lookahead);
    free(env->road_scales);
    free(env->config.control_points);
    free(env->config.lookahead_distances);
    free(env->cars);
    free(env);
}

RACING_API int32_t race_observation_size(const RacingEnv *env)
{
    return 4 * env->config.lookahead_count + 4;
}

static void write_observation(
    const RacingEnv *env, const CarState *car, const GridCell *cell, float *obs)
{
    const RacingConfig *config = &env->config;
    int32_t count = config->lookahead_count;
    const float *targets = &env->map.lookahead[(size_t)cell->row * count * 4];
    float c = cosf(car->heading), s = sinf(car->heading);
    for (int32_t k = 0; k < count * 4; k += 2)
    {
        int32_t horizon = k / 4;
        float dx = targets[k] - car->x;
        float dy = targets[k + 1] - car->y;
        float x = c * dx + s * dy;
        float y = -s * dx + c * dy;
        obs[k] = (x - config->lookahead_distances[horizon]) * env->road_scales[2 * horizon];
        obs[k + 1] = y * env->road_scales[2 * horizon + 1];
    }
    obs[count * 4] = car->forward_speed * env->speed_scale;
    obs[count * 4 + 1] = car->lateral_speed * env->lateral_scale;
    obs[count * 4 + 2] = car->yaw_rate * env->yaw_scale;
    obs[count * 4 + 3] = car->steering_angle * env->steering_scale;
}

RACING_API int32_t race_reset(
    RacingEnv *env,
    int32_t index,
    float *observations)
{
    assert(env && observations);
    assert(index == -1 || (index >= 0 && index < env->num_envs));

    int32_t first = index < 0 ? 0 : index;
    int32_t last = index < 0 ? env->num_envs : index + 1;
    const RacingConfig *config = &env->config;
    const CompiledMap *map = &env->map;
    int32_t obs_size = race_observation_size(env);

    for (int32_t i = first; i < last; i++)
    {
        env->rng ^= env->rng << 13;
        env->rng ^= env->rng >> 7;
        env->rng ^= env->rng << 17;
        float progress = (float)(env->rng >> 40) * (1.0f / 16777216.0f) * map->lap_length;
        env->rng ^= env->rng << 13;
        env->rng ^= env->rng >> 7;
        env->rng ^= env->rng << 17;
        float margin = 0.5f * config->road_width - config->cell_size;
        if (margin < 0.0f)
            margin = 0.0f;
        float offset = ((float)(env->rng >> 40) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * margin;

        int32_t low = 0, high = map->segment_count;
        while (low + 1 < high)
        {
            int32_t middle = low + (high - low) / 2;
            if (map->segments[middle].start_s <= progress)
                low = middle;
            else
                high = middle;
        }

        const RoadSegment *segment = &map->segments[low];
        float u = (progress - segment->start_s) / segment->length;
        CarState *car = &env->cars[i];
        *car = (CarState){0};
        car->next_lap_progress = map->lap_length;
        car->x = segment->ax + u * segment->dx;
        car->y = segment->ay + u * segment->dy;
        car->x += offset * map->normals[2 * low];
        car->y += offset * map->normals[2 * low + 1];
        car->heading = atan2f(segment->dy, segment->dx);

        const GridCell *cell = road_cell(map, car->x, car->y);
        assert(cell);
        car->last_s = cell->progress;
        write_observation(env, car, cell, &observations[(size_t)i * obs_size]);
    }
    return 0;
}

static float calculate_reward(
    const RacingEnv *env,
    float progress_change)
{
    return progress_change / env->map.lap_length;
}

RACING_API void race_step(
    RacingEnv *env,
    const float *actions,
    float *observations,
    float *rewards,
    uint8_t *terminated,
    uint8_t *truncated,
    float *final_observations,
    float *final_progress,
    uint8_t *lap_events,
    float *lap_times)
{
    assert(env && actions && observations && rewards && final_progress && terminated && truncated);
    const RacingConfig *config = &env->config;
    const CompiledMap *map = &env->map;
    int32_t obs_size = race_observation_size(env);
    float dt = config->step_dt / config->substeps;

    for (int32_t i = 0; i < env->num_envs; i++)
    {
        CarState *car = &env->cars[i];
        if (lap_events)
            lap_events[i] = 0;
        if (lap_times)
            lap_times[i] = 0.0f;
        const GridCell *cell = NULL;
        int32_t off_road = 0;
        for (int32_t step = 0; step < config->substeps; step++)
        {
            advance_car(car, config, actions[2 * i], actions[2 * i + 1], dt);
            cell = isfinite(car->x) && isfinite(car->y) ? road_cell(map, car->x, car->y) : NULL;
            if (!cell || !isfinite(car->forward_speed) || !isfinite(car->lateral_speed) || !isfinite(car->yaw_rate) || !isfinite(car->front_slip) || !isfinite(car->rear_slip))
            {
                off_road = 1;
                break;
            }
        }

        float change = 0.0f;
        if (!off_road)
        {
            change = cell->progress - car->last_s;
            if (change > map->lap_length * 0.5f)
                change -= map->lap_length;
            if (change < -map->lap_length * 0.5f)
                change += map->lap_length;
            car->last_s = cell->progress;
            car->progress += change;
        }

        car->steps++;
        if (!off_road && car->progress >= car->next_lap_progress)
        {
            if (lap_events)
                lap_events[i] = 1;
            if (lap_times)
                lap_times[i] = (car->steps - car->last_lap_step) * config->step_dt;
            car->last_lap_step = car->steps;
            car->next_lap_progress += map->lap_length;
        }
        terminated[i] = (uint8_t)off_road;
        truncated[i] = (uint8_t)(!terminated[i] && config->max_steps > 0 && car->steps >= config->max_steps);
        rewards[i] = calculate_reward(env, change);

        if (terminated[i] || truncated[i])
        {
            final_progress[i] = car->progress / map->lap_length;
            if (truncated[i] && final_observations)
                write_observation(env, car, cell, &final_observations[(size_t)i * obs_size]);
            race_reset(env, i, observations);
            continue;
        }

        write_observation(env, car, cell, &observations[(size_t)i * obs_size]);
    }
}
