// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_CPU_TILE_H
#define PBRT_CPU_TILE_H

#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

static const Point2i offsets[] = {Point2i(0, 0), Point2i(0, -1), Point2i(1, 0), 
                                  Point2i(0, 1), Point2i(-1, 0)};

enum EPosition {
    ECur = 0,
    EUp = 1,
    ERight = 2,
    EDown = 3,
    ELeft = 4,
    TILE_SIZE = 5
};

struct MCMCPixel {
    Point2i pos;
    SampledSpectrum values = SampledSpectrum(0.f);
    SampledSpectrum valuesMC = SampledSpectrum(0.f);
};

class MCMCTile {
    public:
    std::vector<MCMCPixel> pixels;
    int nbSamples = 0;
    Float normalization;
    int nbSamplesUni = 0;
    int nbSmallMut = 0;
    int nbSmallMutAcc = 0;
    
    Float scale;
    int REAcc = 0;
    int REAttempt = 0;
    
    Float cumulativeWeight;
    std::vector<SampledSpectrum> current;
    Float impCurr = 0.f;
    SampledWavelengths lambdaCurr;
    
    std::vector<SampledSpectrum> proposed;
    Float impProp = 0.f;
    SampledWavelengths lambdaProp;
    
    Point2i pos;
    
    MLTSampler* sampler;
    
    MCMCTile(const Point2i &imgPos, MLTSampler* sampler) : sampler(sampler), 
    pos(imgPos), 
    pixels(TILE_SIZE, MCMCPixel()) {
        
        // Pixels statistics
        pixels[EPosition::ECur].pos = Point2i(imgPos);
        pixels[EPosition::EUp].pos = Point2i(imgPos + Point2i(0, -1));
        pixels[EPosition::ERight].pos = Point2i(imgPos + Point2i(1, 0));
        pixels[EPosition::EDown].pos = Point2i(imgPos + Point2i(0, 1));
        pixels[EPosition::ELeft].pos = Point2i(imgPos + Point2i(-1, 0));
        
        // The rest of statistics
        scale = 1.f;
        normalization = 0.f;
        nbSamplesUni = 0;
        nbSamples = 0;
        
        // Initialization statistics
        cumulativeWeight = 0.0;
        impCurr = 0.f;
        impProp = 0.f;
        current = std::vector<SampledSpectrum>(TILE_SIZE, SampledSpectrum(0.f));
        proposed = std::vector<SampledSpectrum>(TILE_SIZE, SampledSpectrum(0.f));
    }
    
    ~MCMCTile() { delete sampler; }

    SampledSpectrum get(int i) const {
        if (i < 0 || i >= TILE_SIZE) {
            return SampledSpectrum(0.f);
        }
        return pixels[i].values * scale;
    }
    
    virtual int getNormSamples() const {
        return nbSamplesUni;
    }
    
    virtual Float getNorm() const {
        if (nbSamplesUni == 0) {
            return 1.f;
        }
        return normalization / (Float)nbSamplesUni;
    }

    void scaleNbSamples() {
        if (nbSamples != 0) {
            applyScale(1.0 / (Float) nbSamples);
        } else {
            applyScale(1.0);
        }
    }
    
    void applyScale(Float v) {
        if (!std::isfinite(v)) {
        } else {
            scale *= v;
        }
    }

    void resetScale() {
        scale = 1.f;
    }

    void Accumulate(const std::vector<SampledSpectrum> &res, const Float factor) {
        for (int i = 0; i < TILE_SIZE; i++) {
            SampledSpectrum sample = res[i] * factor;
            pixels[i].values += sample;
        }
    }

    void AccumulateNorm(Float imp, const std::vector<SampledSpectrum>& v) {
        normalization += imp;
        nbSamplesUni += 1;
        for (size_t i = 0; i < TILE_SIZE; i++) {
            pixels[i].valuesMC += v[i];
        }
    }
    
    void Accept(const Float proposedWeight) {
        if (cumulativeWeight != 0.f && impCurr != 0.f){
            Accumulate(current, cumulativeWeight / impCurr);
        }
        
        cumulativeWeight = proposedWeight;
        impCurr = impProp;
        lambdaCurr = lambdaProp;
        impProp = 0.f;
        
        for (int i = 0; i < TILE_SIZE; i++) {
            current[i] = proposed[i];
            proposed[i] = SampledSpectrum(0.f);
        }
        // current = proposed;
        // proposed.clear();
        
        sampler->Accept();
    }
    
    void Reject(const Float proposedWeight) {
        if (proposedWeight != 0.f && impProp != 0.f){
            Accumulate(proposed, proposedWeight / impProp);
        }
        sampler->Reject();
    }
    
    void Flush() {
        if (impCurr != 0.f && cumulativeWeight != 0.f) {
            Accumulate(current, cumulativeWeight / impCurr);
            cumulativeWeight = 0.f;
        }
    }

    void ResetProposed() {
        impProp = 0.f;
        proposed = std::vector<SampledSpectrum>(TILE_SIZE, SampledSpectrum(0.f));
    }

    Float c(int i, SampledWavelengths lambda) const {
        return pixels[i].values.y(lambda);
    }

    // Average luminance of the entire
    Float c(SampledWavelengths lambda) const {
        Float lumSum = 0.0;

        for (MCMCPixel p : pixels)
            lumSum += p.values.y(lambda) * scale;

        lumSum /= pixels.size();

        return lumSum;
    }

    // Returns the 2D point of a pixel via its internal index
    Point2i pixel(int i) const {
        switch (i) {
            case 0:
                return pos;
            case 1:
                return Point2i(pos.x, pos.y - 1);
            case 2:
                return Point2i(pos.x + 1, pos.y);
            case 3:
                return Point2i(pos.x, pos.y + 1);
            case 4:
                return Point2i(pos.x - 1, pos.y);
            default:
                LOG_ERROR("Bad position");
        }
        CHECK(false);
        return Point2i(-1, -1);
    }
};

// Helper function for getting the tildes for a given op
inline int getIMpos(const Bounds2i &imgSize, const Point2i &pos) {
    if (pos.x < imgSize.pMin.x || pos.x >= imgSize.pMax.x || \
        pos.y < imgSize.pMin.y || pos.y >= imgSize.pMax.y)
        return -1; // Invalid ID
        
    return pos.y * imgSize.pMax.x + pos.x;
}
    
inline MCMCTile &getTile(std::vector<MCMCTile *> &tiles, const Bounds2i &imgSize,
                          const Point2i &pos) {
    return *tiles[getIMpos(imgSize, pos)];
}

}  // namespace pbrt

#endif  // PBRT_CPU_SOLVERS_H