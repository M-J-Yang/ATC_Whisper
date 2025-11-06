/**
 * 主应用组件
 */

import React, { useState } from 'react';
import { Layout, Typography, Row, Col, Space } from 'antd';
import ModelConfig from './components/ModelConfig';
import SingleInference from './components/SingleInference';
import RealtimeRecognition from './components/RealtimeRecognition';
import ResultsLog from './components/ResultsLog';
import './App.css';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

function App() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [results, setResults] = useState([]);

  const handleConfigChange = (loaded) => {
    setModelLoaded(loaded);
  };

  const handleResultAdd = (result) => {
    setResults((prev) => [result, ...prev]);
  };

  const handleClearResults = () => {
    setResults([]);
  };

  return (
    <Layout className="app-layout">
      <Header className="app-header">
        <Title level={2} style={{ color: 'white', margin: 0 }}>
          ATC 语音识别系统
        </Title>
      </Header>

      <Content className="app-content">
        <div className="content-wrapper">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {/* 第一行：模型配置 */}
            <Row gutter={[16, 16]}>
              <Col xs={24}>
                <ModelConfig onConfigChange={handleConfigChange} />
              </Col>
            </Row>

            {/* 第二行：推理功能 */}
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <SingleInference
                  modelLoaded={modelLoaded}
                  onResultAdd={handleResultAdd}
                />
              </Col>
              <Col xs={24} lg={12}>
                <RealtimeRecognition
                  modelLoaded={modelLoaded}
                  onResultAdd={handleResultAdd}
                />
              </Col>
            </Row>

            {/* 第三行：结果记录 */}
            <Row gutter={[16, 16]}>
              <Col xs={24}>
                <ResultsLog results={results} onClear={handleClearResults} />
              </Col>
            </Row>
          </Space>
        </div>
      </Content>

      <Footer style={{ textAlign: 'center' }}>
        ATC 语音识别系统 ©2024
      </Footer>
    </Layout>
  );
}

export default App;
