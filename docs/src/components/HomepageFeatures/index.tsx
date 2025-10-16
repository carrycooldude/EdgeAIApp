import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Real AI Inference',
    Svg: require('@site/static/img/edgeai-mountain.svg').default,
    description: (
      <>
        EdgeAI runs actual Llama3.2-1B model inference on your Android device,
        powered by ExecuTorch and Qualcomm QNN acceleration.
      </>
    ),
  },
  {
    title: 'Hardware Accelerated',
    Svg: require('@site/static/img/edgeai-tree.svg').default,
    description: (
      <>
        Leverage Qualcomm's AI Engine Direct with v79 context binaries
        for optimal performance on Snapdragon processors.
      </>
    ),
  },
  {
    title: 'On-Device Processing',
    Svg: require('@site/static/img/edgeai-react.svg').default,
    description: (
      <>
        Process AI requests locally without internet connectivity,
        ensuring privacy and reducing latency for better user experience.
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
