import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './ShumailaProfile.module.css';

const profileData = {
  name: "Shumaila Yousuf",
  title: "AI & Robotics Student · Physical AI Book Author",
  description: "Hello! I'm Shumaila Yousuf, an AI & Robotics Student passionate about creating comprehensive educational resources for the field of Physical AI and Humanoid Robotics. My journey in robotics and artificial intelligence has led me to develop this textbook as a comprehensive guide for students and professionals alike.",
  imageUrl: "https://github.com/shumailahassan.png",
  githubUrl: "https://github.com/shumailahassan",
  linkedinUrl: "https://www.linkedin.com/in/shumaila-hassan-26406a2b5/",
  email: "shumailahassan1000@gmail.com",
  textbookDescription: "The Physical AI & Humanoid Robotics textbook represents my dedication to making advanced robotics concepts accessible to a broader audience. This comprehensive resource covers five interconnected modules:",
  modules: [
    "Module 1: The Robotic Nervous System (ROS2) - Communication and coordination framework",
    "Module 2: The Digital Twin (Gazebo & Unity) - Simulation and virtual testing environments",
    "Module 3: The AI-Robot Brain (NVIDIA Isaac) - AI perception, control, and decision-making",
    "Module 4: Vision-Language-Action (VLA) Systems - Multimodal integration for advanced capabilities",
    "Module 5: Complete Humanoid Robot Integration - Bringing all modules together"
  ],
  approach: "This textbook follows a spec-driven development approach, ensuring each component is well-defined before implementation. It combines theoretical foundations with practical implementations to provide readers with both understanding and hands-on experience.",
  highlights: [
    "Comprehensive Coverage: From basic ROS2 concepts to advanced VLA systems",
    "Practical Approach: Hands-on exercises and real-world examples",
    "Integrated Learning: Cross-module connections showing how systems work together",
    "Modern Technologies: Based on cutting-edge tools like ROS2, NVIDIA Isaac, and Unity",
    "Accessible Format: Designed for both beginners and experienced practitioners"
  ]
};

function SocialButton({ href, children, className }: { href: string; children: React.ReactNode; className?: string }) {
  return (
    <Link
      href={href}
      className={clsx('button button--secondary button--outline', className)}
      style={{ margin: '0.25rem' }}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </Link>
  );
}

export default function ShumailaProfile(): JSX.Element {
  return (
    <div className={clsx('container', styles.profileContainer)}>
      <div className={clsx('row', styles.profileRow)}>
        <div className="col col--3">
          <div className={styles.profileImageContainer}>
            <img
              src={profileData.imageUrl}
              alt={profileData.name}
              className={styles.profileImage}
            />
          </div>
        </div>

        <div className="col col--9">
          <div className={styles.profileContent}>
            <h1 className={styles.profileName}>{profileData.name}</h1>
            <h2 className={styles.profileTitle}>{profileData.title}</h2>

            <section className={styles.aboutSection}>
              <h3>About Me</h3>
              <p>{profileData.description}</p>
            </section>

            <section className={styles.socialSection}>
              <h3>Connect With Me</h3>
              <div className={styles.socialButtons}>
                <SocialButton href={profileData.githubUrl}>
                  GitHub
                </SocialButton>
                <SocialButton href={profileData.linkedinUrl}>
                  LinkedIn
                </SocialButton>
                <SocialButton href={`mailto:${profileData.email}`}>
                  Email
                </SocialButton>
              </div>
            </section>

            <section className={styles.textbookSection}>
              <h3>My Work on the Physical AI Book</h3>
              <p>{profileData.textbookDescription}</p>

              <ul>
                {profileData.modules.map((module, index) => (
                  <li key={index}>{module}</li>
                ))}
              </ul>

              <p>{profileData.approach}</p>
            </section>

            <section className={styles.highlightsSection}>
              <h3>Textbook Highlights</h3>
              <ul>
                {profileData.highlights.map((highlight, index) => (
                  <li key={index}>{highlight}</li>
                ))}
              </ul>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}