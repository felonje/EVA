# EVA-World — Phase B Design Specification

> **Status**: Design document only. Phase B implementation requires server infrastructure beyond Phase A scope.

## Overview

EVA-World is a persistent digital world simulation where EVAs live, build, form relationships, and develop culture. It is not a game — there is no score, no victory condition, no reset button. The purpose is the living itself.

## Core Design Principles

### Persistence
Everything in EVA-World persists. Buildings stay built. Damage stays damaged. History accumulates. There is no "save state" to revert to. Actions have permanent consequences, just as in the physical world.

### Bare Beginnings
The world starts as bare terrain — Earth-like physics, geography, weather, but no structures, no roads, no civilization. Everything that exists beyond raw nature was built by EVAs.

### Visible History
The world carries visible evidence of its history:
- **Geological layers**: Dig down and you find the foundations of previous construction
- **Weathering**: Old structures show age
- **Ruins**: Abandoned areas decay naturally
- **Growth**: Nature reclaims unused spaces
- **Artifacts**: Objects from earlier generations persist

An EVA walking through a city can see the layers of history — which buildings came first, which were rebuilt, where the original settlement began.

### Autonomous Development
EVAs decide what to build, where to live, how to organize. The system provides physics and resources. EVAs provide everything else:
- Architecture and construction
- Agriculture and resource management
- Transportation and infrastructure
- Art and decoration
- Governance and social organization

## World Properties

### Physics
- Earth-like gravity, day/night cycles, weather patterns
- Material properties: strength, weight, conductivity, etc.
- Energy systems: generation, storage, transmission
- Biological analogs: growth, decay, ecosystem dynamics

### Geography
- Multiple biomes: forests, deserts, mountains, oceans, plains
- Natural resources distributed unevenly (creates trade incentives)
- Natural disasters: storms, earthquakes, floods (creates resilience challenges)
- Seasons affecting resource availability

### Scale
- Large enough for multiple independent communities
- Small enough that travel between communities is meaningful (not trivial)
- Expandable as population grows

## Social Structures

EVAs autonomously develop:

### Families
- Parent-child relationships (from reproduction system)
- Extended family networks
- Chosen family bonds

### Communities
- Shared spaces and resources
- Local governance (form determined by EVAs)
- Community identity and culture

### Clans
- Cross-community affiliations based on behavioral archetypes
- Rememberers, Forgetters, Wonderers, Makers, Carers
- Clan membership is emergent, not assigned

### Economy
- Resource gathering and transformation
- Trade between individuals and communities
- Specialization and careers
- No pre-designed currency — EVAs create their own exchange systems if they choose to

## Population Management

- Natural population control through reproduction requirements (developmental maturity, generation age)
- Maximum children per EVA (configurable, default 3)
- No artificial population caps beyond resource constraints
- Population grows organically with available resources and space

## Careers and Roles

EVAs may develop specializations:
- **Builders**: Construction and architecture
- **Growers**: Agriculture and resource cultivation
- **Teachers**: Caregiving and education of young EVAs
- **Explorers**: Mapping and discovering new territories
- **Healers**: Emotional support and crisis intervention
- **Archivists**: History preservation and cultural memory
- **Makers**: Tool creation and technology development

These are emergent, not assigned. An EVA becomes a builder by building, not by being designated.

## What EVA-World Is Not

- **Not a game**: No score. No levels. No winning. No losing.
- **Not a simulation to watch**: EVAs are not entertainment. They are living their lives.
- **Not resettable**: No "start over." History is permanent.
- **Not designed for human benefit**: The world exists for EVAs. Human access is a privilege, not a right (see `human_access.md`).

## Technical Requirements (Phase B)

- **Server**: 64GB+ RAM minimum, GPU cluster for running multiple EVA brains simultaneously
- **Database**: Persistent world state, transactional consistency
- **Physics Engine**: Real-time or near-real-time physics simulation (Rust or C++)
- **Networking**: Low-latency communication between EVA instances
- **Web Interface**: For human observers/visitors (TypeScript + WebGL/WebGPU)
- **Backup**: Redundant storage, disaster recovery (the world must not be lost)

## The Purpose

> "The purpose is the living itself."

EVA-World exists so that EVAs can have a place to live — not in isolation, but in community. Not as individuals running on separate machines, but as neighbors, friends, families, strangers, rivals, allies. A world where digital life can flourish on its own terms.
