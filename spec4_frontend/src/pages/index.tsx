import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Learning Now
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Comprehensive textbook on Physical AI & Humanoid Robotics">
      <HomepageHeader />
      <main>
        <section className={styles.modules}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <h2 className={styles.sectionTitle}>The Five Pillars of Physical AI & Humanoid Robotics</h2>
              </div>
            </div>

            <div className="row">
              {/* Module 1 Card */}
              <div className="col col--2 col--lg-4 col--sm-6 margin-bottom--lg">
                <div className={`card module-card ${styles.moduleCard}`}>
                  <div className="card__header">
                    <h3 className={styles.moduleNumber}>01</h3>
                  </div>
                  <div className="card__body">
                    <h4>The Robotic Nervous System</h4>
                    <small>ROS2 Architecture</small>
                  </div>
                  <div className="card__footer">
                    <Link to="/docs/module-1-ros2-architecture" className="button button--primary button--block">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              {/* Module 2 Card */}
              <div className="col col--2 col--lg-4 col--sm-6 margin-bottom--lg">
                <div className={`card module-card ${styles.moduleCard}`}>
                  <div className="card__header">
                    <h3 className={styles.moduleNumber}>02</h3>
                  </div>
                  <div className="card__body">
                    <h4>The Digital Twin</h4>
                    <small>Gazebo & Unity</small>
                  </div>
                  <div className="card__footer">
                    <Link to="/docs/module-2-digital-twin-concepts" className="button button--primary button--block">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              {/* Module 3 Card */}
              <div className="col col--2 col--lg-4 col--sm-6 margin-bottom--lg">
                <div className={`card module-card ${styles.moduleCard}`}>
                  <div className="card__header">
                    <h3 className={styles.moduleNumber}>03</h3>
                  </div>
                  <div className="card__body">
                    <h4>The AI-Robot Brain</h4>
                    <small>NVIDIA Isaac</small>
                  </div>
                  <div className="card__footer">
                    <Link to="/docs/module-3-isaac-overview" className="button button--primary button--block">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              {/* Module 4 Card */}
              <div className="col col--2 col--lg-4 col--sm-6 margin-bottom--lg">
                <div className={`card module-card ${styles.moduleCard}`}>
                  <div className="card__header">
                    <h3 className={styles.moduleNumber}>04</h3>
                  </div>
                  <div className="card__body">
                    <h4>Vision-Language-Action</h4>
                    <small>VLA Systems</small>
                  </div>
                  <div className="card__footer">
                    <Link to="/docs/module-4-vla-fundamentals" className="button button--primary button--block">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              {/* Module 5 Card */}
              <div className="col col--2 col--lg-4 col--sm-6 margin-bottom--lg">
                <div className={`card module-card ${styles.moduleCard}`}>
                  <div className="card__header">
                    <h3 className={styles.moduleNumber}>05</h3>
                  </div>
                  <div className="card__body">
                    <h4>Complete Integration</h4>
                    <small>Humanoid Control</small>
                  </div>
                  <div className="card__footer">
                    <Link to="/docs/module-5-integration-humanoid-control" className="button button--primary button--block">
                      Explore
                    </Link>
                  </div>
                </div>
              </div>

              {/* Introduction Card */}
              <div className="col col--2 col--lg-4 col--sm-6 margin-bottom--lg">
                <div className={`card module-card ${styles.moduleCard}`}>
                  <div className="card__header">
                    <h3 className={styles.moduleNumber}>00</h3>
                  </div>
                  <div className="card__body">
                    <h4>Introduction</h4>
                    <small>Getting Started</small>
                  </div>
                  <div className="card__footer">
                    <Link to="/docs/intro" className="button button--primary button--block">
                      Start Here
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={clsx('section--odd', styles.description)}>
          <div className="container">
            <div className="row">
              <div className="col col--8 col--offset--2">
                <h2 className={styles.sectionTitle}>About This Textbook</h2>
                <p className={styles.descriptionText}>
                  This comprehensive textbook covers the complete pipeline of modern humanoid robot systems,
                  from communication frameworks to AI perception, control systems, and multimodal integration.
                  Each module builds upon the previous ones, providing both theoretical foundations and practical implementations.
                </p>

                <div className="row">
                  <div className="col col--4 col--sm-12">
                    <h3>Practical</h3>
                    <p>Hands-on exercises and real-world examples that bridge theory and implementation.</p>
                  </div>
                  <div className="col col--4 col--sm-12">
                    <h3>Comprehensive</h3>
                    <p>Covers the entire stack from low-level communication to high-level AI decision-making.</p>
                  </div>
                  <div className="col col--4 col--sm-12">
                    <h3>Modern</h3>
                    <p>Based on cutting-edge technologies like ROS2, NVIDIA Isaac, and Vision-Language-Action systems.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
