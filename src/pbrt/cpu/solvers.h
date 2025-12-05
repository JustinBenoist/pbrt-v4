// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_SOLVERS_H
#define PBRT_CPU_SOLVERS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/cpu/tile.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/progressreporter.h>

#include <functional>
#include <memory>
#include <ostream>
#include <iostream>
#include <string>
#include <vector>

namespace pbrt {

#define SUM_POS 1

class TileIterativeReweighted {
    int dumpID = 0;

protected:
    Vector2i m_imgSize;
    std::vector<MCMCTile*> &m_tiles;
    const int nbIter;
    const Float alpha = 0.05;
    const bool errorWeights = true;

public:
struct TileCache {
    Float values[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    Float sum = 0.0;
    Float avgNonNull = 0.0;
    Float sumMC = 0.0;

    void Initialize(MCMCTile* tile, const Film &film, int channel, 
                    const std::vector<RGB> &mc_estimate, const Vector2i &imgSize) {
        int nbNonNull = 0;
        sum = 0;
        sumMC = 0;

        for (int i = 0; i < 5; i++) {
            values[i] = film.ToOutputRGB(tile->get(i), film.SampleWavelengths(0.5f))[channel];
            Point2i pixel = tile->pixel(i);

            // Compute MC value if inside image
            Float valueMC = 0.0f;
            if (pixel.x >= 0 && pixel.x < imgSize.x &&
                pixel.y >= 0 && pixel.y < imgSize.y) {
                int bufferIndex = pixel.y * imgSize.x + pixel.x;
                valueMC = mc_estimate[bufferIndex][channel];
            }

            // Accumulate only positive values
            if (values[i] > 0.0f) {
                sum += values[i];
                sumMC += valueMC;
                nbNonNull++;
            }
        }

        // Safely compute averages
        if (nbNonNull > 0) {
            const Float inv = 1.0f / nbNonNull;
            avgNonNull = sum * inv;
            sum = sum * inv;
            sumMC = sumMC * inv;
        } 
        else {

            avgNonNull = 0.0f;
            sum = 0.0f;
            sumMC = 0.0f;
        }
    }

    float operator[](size_t index) const { return values[index]; }
};


public:
    TileIterativeReweighted(const Bounds2i &imgBounds, std::vector<MCMCTile *> &tiles, 
                            int nbIter) : m_tiles(tiles), nbIter(nbIter)  {
        m_imgSize = {imgBounds.pMax.x, imgBounds.pMax.y};
    }

    std::vector<SampledSpectrum> MCEstimates() {
        std::vector<SampledSpectrum> accum(m_imgSize.x * m_imgSize.y, SampledSpectrum(0.f));
        int *sampleCounts = new int[m_imgSize.x * m_imgSize.y]();

        // Push each of the tiles to the buffer
        ParallelFor(0, m_tiles.size(), [&](int i) {
            MCMCTile* tile = m_tiles[i];
            int sampleCount = tile->nbSamplesUni;

            for (size_t pixelIndex = 0; pixelIndex < TILE_SIZE; ++pixelIndex) {
                SampledSpectrum normalizedValue = tile->pixels[pixelIndex].valuesMC;

                Point2i pixelLocation = tile->pixel(pixelIndex);
                if (pixelLocation.x < 0 || pixelLocation.x >= m_imgSize.x || pixelLocation.y < 0
                    || pixelLocation.y >= m_imgSize.y)
                    continue;

                int bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
                accum[bufferIndex] += normalizedValue;
                sampleCounts[bufferIndex] += sampleCount;
            }
        });

        // Scale the accumulted flux by the sample counts
        for (size_t i = 0, yy = 0; yy < m_imgSize.y; ++yy) {
            for (size_t xx = 0; xx < m_imgSize.x; ++xx, ++i) {
                if (sampleCounts[i] == 0.f) {
                    accum[i] = SampledSpectrum(0.f);
                } else {
                    accum[i] /= (Float)sampleCounts[i];
                }
            }
        }
        delete[] sampleCounts;
        return accum;
    }

    void solveScaling(std::vector<Float> &b, const Film& film, int channel,
                      const std::vector<RGB> &mcEstimate) {
        const bool DIAG_USE = true;//!m_config.REWeighting;
        const bool ONE_OVERLAP = true;//!m_config.REWeighting;
        const bool ATTACH_TILE = true;
        std::vector<Float> w(m_imgSize.x*m_imgSize.y, 1.0);

        std::vector<TileCache> lum_buff(m_imgSize.x * m_imgSize.y);
        for (size_t y = 0; y < m_imgSize.y; y++) {
            for (size_t x = 0; x < m_imgSize.x; x++) {
                size_t t = y * m_imgSize.x + x;
                MCMCTile *curr_tile = m_tiles[t];
                lum_buff[t].Initialize(curr_tile, film, channel, mcEstimate, m_imgSize);

                b[t] = curr_tile->getNorm();
            }
        }
        std::cout << "Solving channel " << channel << "\n";
        // std::vector<Float> b0 = b;

        std::vector<Float> b_next(m_imgSize.x * m_imgSize.y, Float(0.f));
        const int EPOCHS = 20;
        ProgressReporter progress(EPOCHS, "Reconstructing", Options->quiet);
        for(auto iter = 0; iter < EPOCHS; iter++) {
            for (size_t t = 0; t < nbIter; t++) {
                std::fill(b_next.begin(), b_next.end(), 0.f);
                // ParallelFor(0, m_tiles.size(), [&](int i) {
                // TODO: Turn this parallel, make sure it doesn't break
                for (size_t y = 0; y < m_imgSize.y; y++) {
                    for (size_t x = 0; x < m_imgSize.x; x++) {
                        // Compute the next step
                        size_t curr_id = y * m_imgSize.x + x;
                        Point2i pixelLocation = m_tiles[curr_id]->pixel(0);
                        // int x = pixelLocation.x;
                        // int y = pixelLocation.y;
                        // int curr_id = pixelLocation.y * m_imgSize.x + pixelLocation.x;
                        const TileCache &curr_tile = lum_buff[curr_id];

                        struct ResultForce {
                            Float force = 0.f;
                            Float pos = 0.f;
                        };
                        ResultForce res = {};

                        auto apply_force = [this](ResultForce &r, Float b1, Float v1, Float b2, 
                                                Float v2, Float w1, Float w2) -> void {
                            // Computes the weights
                            Float w = std::min(w1,w2);//std::max(w1, w2);
                            if(!errorWeights) {
                                w = 1.0;
                            }
                            // if(m_config.customWeights) {
                            //     Float w_c = std::min(v2 / v1, v1 / v2);
                            //     w_c *= w_c;
                            //     w *= w_c;
                            // }

                            // function f
                            Float f = 0.5 * (v1 * b1 - v2 * b2);
                            if (std::isfinite(f) && v1 != 0.0 && v2 != 0.0) {
                                r.force += w * f;
    #if SUM_POS
                                r.pos += w * v1;
    #else
                                r.pos += w;
    #endif
                            }
                        };

                        if (curr_tile.sum == 0.0) {
                            b_next[curr_id] = b[curr_id];
                            continue;
                        }
                        if (ATTACH_TILE) {
                            Float force = b[curr_id] * lum_buff[curr_id].sum - lum_buff[curr_id].sumMC;
                            Float weight = alpha * w[curr_id];
                            res.force += weight * force;
    #if SUM_POS
                            res.pos += weight * lum_buff[curr_id].sum;
    #else
                            res.pos += weight * 5;
    #endif
                        }

                        // Left
                        if (x != 0) {
                            size_t next_id = curr_id - 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECur],
                                        b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[ELeft],
                                        b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }
                        // Right
                        if (x != m_imgSize.x - 1) {
                            size_t next_id = curr_id + 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECur],
                                        b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[ERight],
                                        b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }
                        // Top
                        if (y != 0) {
                            size_t next_id = curr_id - m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECur],
                                        b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[EDown],
                                        b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }
                        // Bottom
                        if (y != m_imgSize.y - 1) {
                            size_t next_id = curr_id + m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            apply_force(res, b[curr_id], curr_tile[ECur],
                                        b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                            apply_force(res, b[curr_id], curr_tile[EUp],
                                        b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }

                        // More overlapping
                        if (DIAG_USE) {
                            if (x != 0 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[ELeft],
                                            b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EDown],
                                            b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[ERight],
                                            b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EUp],
                                            b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            }
                            if (x != 0 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[ELeft],
                                            b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EUp],
                                            b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[ERight],
                                            b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                                apply_force(res, b[curr_id], curr_tile[EDown],
                                            b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            }
                        }

                        if (ONE_OVERLAP) {
                            if (x > 1) {
                                size_t next_id = curr_id - 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[ELeft],
                                            b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            }

                            // Right
                            if (x < m_imgSize.x - 2) {
                                size_t next_id = curr_id + 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[ERight],
                                            b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            }

                            // Top
                            if (y > 1) {
                                size_t next_id = curr_id - 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EDown],
                                            b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                            }

                            // Bottom
                            if (y < m_imgSize.y - 2) {
                                size_t next_id = curr_id + 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                apply_force(res, b[curr_id], curr_tile[EUp],
                                            b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                            }
                        }

                        if (res.pos != 0.0) {
                            // average forces * 0.5 -> gives how much the tile move
                            // dividing by tile luminance average give back the normalization factor?
    #if SUM_POS
    #else
                            res.pos *= lum_buff[curr_id].avgNonNull;
    #endif
                            Float new_value = b[curr_id] - (res.force / res.pos);
                            b_next[curr_id] = new_value;
                        } else {
                            b_next[curr_id] = b[curr_id];
                        }
                        b_next[curr_id] = std::max(b_next[curr_id], Float(0.0));
                    }
                }
                // Optimization from Rex for the coforce
                auto nbNegativeNormalization = 0;
                for (size_t k = 0; k < m_imgSize.x * m_imgSize.y; k++) {
                    if (b_next[k] <= 0.0) {
                        nbNegativeNormalization += 1;
                    }
                    b[k] = b_next[k];
                }
            }

            if(errorWeights){
                // Compute the error
                std::vector<Float> w2 = w;
                for (size_t y = 0; y < m_imgSize.y; y++) {
                    for (size_t x = 0; x < m_imgSize.x; x++) {
                        // TODO: Same thing, make it parallel
                        // Compute the next step
                        size_t curr_id = y * m_imgSize.x + x;
                        const TileCache &curr_tile = lum_buff[curr_id];
                        Float error = 0.0;

                        if (ATTACH_TILE) {
                            Float force = b[curr_id] * lum_buff[curr_id].sum - lum_buff[curr_id].sumMC;
                            Float weight = alpha; //* w[curr_id];
                            error += weight * std::abs(force);
                        }

                        auto add_error = [this](Float &r, Float b1, Float v1, Float b2, Float v2, Float w1, Float w2) -> void {
                            Float w = 1.0;
                            // if(m_config.customWeights) {
                            //     Float w_c = std::min(v2 / v1, v1 / v2);
                            //     w_c*= w_c;
                            //     w *= w_c;
                            // }
                            Float f = 0.5 * (v1 * b1 - v2 * b2);
                            if (std::isfinite(f) && v1 != 0.0 && v2 != 0.0) {
                                r += std::abs(f) * w;
                            }
                        };

                        if (x != 0) {
                            size_t next_id = curr_id - 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECur],
                                      b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[ELeft],
                                      b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }
                        // Right
                        if (x != m_imgSize.x - 1) {
                            size_t next_id = curr_id + 1;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECur],
                                      b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[ERight],
                                      b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }
                        // Top
                        if (y != 0) {
                            size_t next_id = curr_id - m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECur],
                                      b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[EDown],
                                      b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }
                        // Bottom
                        if (y != m_imgSize.y - 1) {
                            size_t next_id = curr_id + m_imgSize.x;
                            const TileCache &next_tile = lum_buff[next_id];
                            add_error(error, b[curr_id], curr_tile[ECur],
                                      b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                            add_error(error, b[curr_id], curr_tile[EUp],
                                      b[next_id], next_tile[ECur], w[curr_id], w[next_id]);
                        }

                        // More overlapping
                        if (DIAG_USE) {
                            if (x != 0 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[ELeft],
                                          b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EDown],
                                          b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[ERight],
                                          b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EUp],
                                          b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            }
                            if (x != 0 && y != m_imgSize.y - 1) {
                                size_t next_id = curr_id + m_imgSize.x - 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[ELeft],
                                          b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EUp],
                                          b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            }
                            if (x != m_imgSize.x - 1 && y != 0) {
                                size_t next_id = curr_id - m_imgSize.x + 1;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[ERight],
                                          b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                                add_error(error, b[curr_id], curr_tile[EDown],
                                          b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            }
                        }

                        if (ONE_OVERLAP) {
                            if (x > 1) {
                                size_t next_id = curr_id - 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[ELeft],
                                          b[next_id], next_tile[ERight], w[curr_id], w[next_id]);
                            }

                            // Right
                            if (x < m_imgSize.x - 2) {
                                size_t next_id = curr_id + 2;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[ERight],
                                          b[next_id], next_tile[ELeft], w[curr_id], w[next_id]);
                            }

                            // Top
                            if (y > 1) {
                                size_t next_id = curr_id - 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EDown],
                                          b[next_id], next_tile[EUp], w[curr_id], w[next_id]);
                            }

                            // Bottom
                            if (y < m_imgSize.y - 2) {
                                size_t next_id = curr_id + 2 * m_imgSize.x;
                                const TileCache &next_tile = lum_buff[next_id];
                                add_error(error, b[curr_id], curr_tile[EUp],
                                          b[next_id], next_tile[EDown], w[curr_id], w[next_id]);
                            }
                        }

                        w2[curr_id] = 1.0 / (error + std::max(0.05*std::pow(0.5, iter), 0.0001));
                    }
                }

                Float sumW2 = 0.0;
                for(auto wc: w2) {
                    sumW2 += wc;
                }
                for(Float& wc: w2) {
                    wc *= (m_imgSize.x * m_imgSize.y) / sumW2;
                }
                w = w2; // Update
            }
            progress.Update();
        }
        progress.Done();
    }

    std::vector<RGB> solve(const Film& film) {
        // Get more robust MC estimates
        // By combining the different uniform estimates
        std::vector<SampledSpectrum> mcEstimateSpectral = MCEstimates();
        std::vector<RGB> mcEstimate(mcEstimateSpectral.size());
        // TODO : change that for spectral rendering 
        for (int i = 0; i < mcEstimate.size(); i++) {
            mcEstimate[i] = film.ToOutputRGB(mcEstimateSpectral[i], film.SampleWavelengths(0.5));
        }

        // Here we will align each tiles color independently.
        // we could also do only luminance based alignment procedure
        // by this pervent us to remove the color noise in the tile's estimates.
        std::vector<Float> bRed(m_imgSize.x * m_imgSize.y, 0.f);
        solveScaling(bRed, film, 0, mcEstimate);
        std::vector<Float> bGreen(m_imgSize.x * m_imgSize.y, 0.f);
        solveScaling(bGreen, film, 1, mcEstimate);
        std::vector<Float> bBlue(m_imgSize.x * m_imgSize.y, 0.f);
        solveScaling(bBlue, film, 2, mcEstimate);

        // Combine all scaling factor to produce the last image
        std::vector<RGB> accumBuffer(mcEstimate.size());
        RGB *accum = (RGB *) accumBuffer.data();
        int *sampleCounts = new int[m_imgSize.x * m_imgSize.y]();
        ParallelFor(0, m_tiles.size(), [&](int k) {
            MCMCTile *tile = m_tiles[k];
            int sampleCount = tile->nbSamples;
            RGB rgbScale = {bRed[k], bGreen[k], bBlue[k]};
            // std::cout << rgbScale << "\n";

            for (size_t pixelIndex = 0; pixelIndex < TILE_SIZE; ++pixelIndex) {
                // TODO: change that for spectral
                RGB pixelValue = film.ToOutputRGB(tile->pixels[pixelIndex].values, 
                                                  film.SampleWavelengths(0.5));
                RGB normalizedValue = pixelValue * rgbScale;

                Point2i pixelLocation = tile->pixel(pixelIndex);
                if (pixelLocation.x < 0 || pixelLocation.x >= m_imgSize.x || pixelLocation.y < 0
                    || pixelLocation.y >= m_imgSize.y)
                    continue;

                int bufferIndex = pixelLocation.y * m_imgSize.x + pixelLocation.x;
                accum[bufferIndex] += normalizedValue;
                sampleCounts[bufferIndex] += sampleCount;
            }

        });

        // Scale the accumulted flux by the sample counts
        for (size_t i = 0, yy = 0; yy < m_imgSize.y; ++yy) {
            for (size_t xx = 0; xx < m_imgSize.x; ++xx, ++i) {
                if (sampleCounts[i] == 0.0) {
                    accum[i] = RGB(0.f, 0.f, 0.f);
                } else {
                    accum[i] /= (Float) sampleCounts[i];
                }
            }
        }
        delete[] sampleCounts;


        return accumBuffer;
    }
};

}  // namespace pbrt

#endif  // PBRT_CPU_SOLVERS_H