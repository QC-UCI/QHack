## Quantum Enhanced GAN for HEP

### Overview 
To enhance the Generative Adversarial Networks (GAN) used in the High Energy Physics (HEP) community for fast event simulation with Quantum Circuit Born Machine (QCBM), a versatile and efficient quantum generative model, to sample the prior (latent space). The quantum enhanced architecture, Quantum Circuit Associative Adversarial Network (QC-AAN), was shown previously to not only have similar performance as DCGAN but also have  practical quantum advantages such as greater training stability on MNIST [1]. Instabilities of the training caused by diverging gradient and vanishing gradient are a major practical concern, especially for the HEP community\*[2]. So, if a QC-AAN can make the training for GANs more robust, we would expect it to have practical value for the HEP community. We plan to build upon CaloGAN [3], a popular architecture to generate HEP detector responses and use vanilla CaloGAN as a baseline for comparison. 

\* To overcome the training instability, HEP community often uses Wasserstein GANs. Due to time constraints, we plan to investigate a quantum enhanced Wassertein GANs in the future.

### Procedure
- Construct QC-AAN with multi-basis QCBM and CaloGAN

- Run experiments against particle physics dataset and compare it against vanilla CaloGAN with the metrics in the next section

- If time permits, repeat the experiments and compare it against 
  - Wasserstein CaloGAN
  - Restricted Boltzmann Machine (RBM) based AAN


### Metrics
- Inception score
- HEP based similarity score
  - 1-D showering statistics
  - Energy flow polynomials (EFPs)
- Training stability
- Mode (energy) diversity


### Resource Estimate



### Reference
[1] M. S. Rudolph, N. B. Toussaint, A. Katabarwa, S. Johri, B. Peropadre, and A. Perdomo-Ortiz, Generation of High-Resolution Handwritten Digits with an Ion-Trap Quantum Computer, (2020).

[2] A. Butter and T. Plehn, Generative Networks for LHC Events, ArXiv:2008.08558 [Hep-Ph] (2020).

[3] M. Paganini, L. de Oliveira, and B. Nachman, CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks, Phys. Rev. D 97, 014021 (2018).

