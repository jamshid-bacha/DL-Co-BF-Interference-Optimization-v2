# Reinforcement Learning based WiFi 8 (IEEE 802.11bn) Beamforming Configuration for Interference Management

Next-generation Wi-Fi networks are expected to have an ultra-dense deployment of access points (APs), thus, interference from overlapping basic service sets (OBSSs) poses challenges for interference management. 
Wi-Fi 8 aims at mitigating such interference using multi-access point coordination (MAPC).
One of the MAPC variants is coordinated beamforming (Co-BF), where neighboring APs direct their signals towards specific users. 
Besides beam steering, APs can also perform null steering, which is more complex but can bring greater performance gains. 
In this paper, we present a centralized approach named intelligent null steering by reinforcement learning (IntelliNull), designed to reduce interference from neighboring transmitters by coordinated nulling while maximizing the signal quality at each station.
We show that training the beam and null steering mechanism with a deep deterministic policy gradient (DDPG), it is possible to steer beams toward associated stations while intelligently nulling the most destructive interference from OBSS rather than nulling random interference directions. 
This method enhances communication between the AP and neighboring stations by reducing channel access contention, enabling transmissions at full power, and reducing worst-case latency. 
The proposed IntelliNull agent continuously adapts to changes in the network environment, including node mobility using channel state information (CSI) collected in real-time. 
We also compare our IntelliNull, which is based on beamforming plus nulling, with the baseline which is based on beamforming only. 
Our results demonstrate that IntelliNull outperforms the baseline by effectively mitigating interference, leading to higher throughput and better signal-to-interference-plus-noise ratio (SINR), especially in dense deployment scenarios where beamforming alone fails to sufficiently suppress OBSS interference.

## Problem Statement
We consider the downlink in a multi-AP scenario where the APs are equipped with a uniform linear array (ULA). The task is to perform Co-BF, i.e. beamform own client while nulling towards most disruptive clients in neighboring OBSS.

Assumptions/limitations:
1. Coordinated TDMA - slots are aligned among APs
2. No carrier sensing and frequency reuse 1 (all cells on the same full channel)

## Approach
We aim to solve the problem with a DDPG. 

## How to run the code
Training DDPG &#8594; python training.py

Testing DDPG &#8594; testing.py

Oracle python &#8594; Oracle.py

### Resources
1. Context DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
2. Stable RL: https://stable-baselines3.readthedocs.io/en/master/index.html

### File Details

| File Name                 | Notes                                        |  Details                                                                                                               |
|---------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| BS.py                     | Base Stations Code                           | Base stations communicating with their respective stations                                                             |
| training.py               | DDPG, Actor, Critic Main Code                | DDPG Agent, Actor, Critic Network, Buffer, and main file to run                                                        |
| STA.py                    | Stations Code                                | Stations configuration and how stations are mobile around the base stations                                            |
| Sim.py                    | Simulation and Step Function Code            | The Simulation file includes collecting all the information with the step function from the whole environment          |
| ULA.py                    | Uniform Linear Array Code                    | Here we designed the Uniform Linear Array Antenna, which follows the Zero Forcing equation for beamforming and nulling |
| config.py                 | Configuration Code                           | This file includes the parameters to use during the simulation e.g. No. of station used, No. of Antennas etc.          |
| helper.py                 | Helper Code                                  | Helper file helps to calculate all the equations and conversion of the math units                                      |


### Wireless Communication Parameters
| Parameters                               | Values                                        
|------------------------------------------|----------------------------------|
| Frequency band                           | 5 GHz                            |
| Channel bandwidth                        | 80 MHz                           | 
| Noise figure                             | 7 dB                             |
| Thermal noise at room temperature        | -174 dBm                         |
| Stations mobility                        | 1.4 m/s                          |
| Time slot                                | 50 ms                            |
| Max. radius station to AP                | 10 m                             |
| CSI acquisition interval                 | 100 ms                           |
| Max. transmit power                      | 20 dBm                           |
| Pathloss model                           | FSPL                             |





# What is Beamforming and Nulling
Beamforming is a process that directs radio waves in a specific direction, like a spotlight, instead of transmitting in all directions. 
Nulling, also known as null-steering, is a technique used to intentionally create areas of low signal strength in specific directions, as shown in the figure below.

<img width="389" alt="only_beamforming" src="https://github.com/user-attachments/assets/30573a43-3a6a-48e5-8bf1-6a7938bf4321" />


<img width="389" alt="beamforming_with_nulling" src="https://github.com/user-attachments/assets/ea003583-8090-48d3-baf5-17ad9452df61" />




# Environment
<img width="791" alt="Simulation_v5" src="https://github.com/user-attachments/assets/2173b9e8-421a-4c41-a907-ad542734ad11" style="width: 800px;">




# Proposed Architecture Diagram

<img width="1052" alt="ArchitictureDiagram" src="https://github.com/user-attachments/assets/26f723b4-2b5a-46a1-8f07-25d034237949" style="width: 800px;">



# DDPG vs Oracle Rewards
<img width="1108" alt="Archticture_Diagram_v3" src="https://github.com/user-attachments/assets/e8ce2ad1-4d53-46f5-83f9-d68d650a7f51" style="width: 800px;">




# Rewards based on Data Rate
### 4 Antennas, 7 Base Stations, 1 Station per BS
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/273a598a-b8bc-4058-bdf9-68caaed6d478" alt="4_Antennass_2_Beam_Angles_2_Null_Angles" style="width: 650px; height: 350px"/>
    </td>
  </tr>
</table>


## 📄 Citation
If you use this repository in your research, please cite our paper:

```bibtex
@article{bacha2025deep,
    author = {Bacha, Jamshid and Zubow, Anatolij and Szott, Szymon and Kosek-Szott, Katarzyna and Dressler, Falko},
    doi = {10.1016/j.comcom.2025.108286},
    title = {{Deep Reinforcement Learning based Interference Optimization for Coordinated Beamforming in Ultra-Dense Wi-Fi Networks}},
    pages = {108286},
    journal = {Elsevier Computer Communications},
    issn = {0140-3664},
    publisher = {Elsevier},
    month = {10},
    volume = {242},
  ` doi = {{https://doi.org/10.1016/j.comcom.2025.108286}},
    year = {2025}
}



